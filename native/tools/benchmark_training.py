"""Same-GPU training-throughput benchmark: native cuda_detector_train vs original pjreddie Darknet and PyTorch.

All sides train the same cfg at a fixed input size with the same per-step minibatch, real VOC2007
images and comparable crop/flip/HSV augmentation. Throughput is steady-state wall-clock images/s:
native and PyTorch (native/tools/pytorch_baseline.py, same CLI, run by `--python`) = images per
epoch / time between consecutive epoch checkpoints (epoch 1 is warm-up); Darknet (in the
`darknet-train:gpu` Docker image, subdivisions=1, random=0) = images between two iteration lines. GPU temperature/clock/power/utilization (nvidia-smi, 1 s) are summarized over the
measured window so throttling is visible; `--cool-to` waits for a cool GPU before starting.
Each run appends one JSON line to `--results`. Data/weights paths follow docs/performance.md.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs" / "benchmarks"
NVSMI = ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,power.draw,utilization.gpu",
         "--format=csv,noheader,nounits", "-lms", "1000"]
EXE = ".exe" if os.name == "nt" else ""

MODELS = {
    "yolov3-tiny": {"cfg": "runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg",
                    "init": "data/yolov3-tiny.weights", "init_layers": 15,
                    "darknet_init": "runs/darknet-docker/yolov3-tiny-voc.conv.15"},
    "yolov2": {"cfg": "runs/yolov2-voc/yolov2-voc.cfg",
               "init": "data/darknet19_448.conv.23", "init_layers": 23,
               "darknet_init": "data/darknet19_448.conv.23"},
    "yolov3": {"cfg": "runs/yolov3-voc/yolov3-voc.cfg",
               "init": "data/darknet53.conv.74", "init_layers": 75,
               "darknet_init": "data/darknet53.conv.74"},
}


class Telemetry:
    """Samples nvidia-smi once per second in a background thread, tagged with perf_counter time."""

    def __init__(self):
        """Starts the nvidia-smi sampling process and its reader thread."""
        self.samples = []
        self.proc = subprocess.Popen(NVSMI, stdout=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        """Parses each nvidia-smi CSV line into a timestamped sample."""
        for line in self.proc.stdout:
            try:
                self.samples.append((time.perf_counter(), *map(float, line.split(","))))
            except ValueError:
                pass

    def stop(self):
        """Terminates the sampling process."""
        self.proc.terminate()

    def window(self, t0, t1):
        """Returns mean SM clock/power/utilization and max temperature over [t0, t1]."""
        rows = [s for s in self.samples if t0 <= s[0] <= t1]
        if not rows:
            return {}
        return {"temp_max": max(r[1] for r in rows), "sm_mhz_mean": statistics.mean(r[2] for r in rows),
                "power_w_mean": statistics.mean(r[3] for r in rows), "util_mean": statistics.mean(r[4] for r in rows)}


def gpu_temperature():
    """Current GPU temperature in C."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def run_streaming(cmd, log_path, line_hook):
    """Runs cmd, timestamps every output line into log_path and calls line_hook(t, line)."""
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("# " + " ".join(map(str, cmd)) + "\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                cwd=ROOT, bufsize=1, errors="replace")
        for line in proc.stdout:
            t = time.perf_counter()
            log.write(f"{t:.3f} {line}")
            line_hook(t, line.rstrip("\n"))
        return proc.wait()


def native(args, model):
    """Times native or PyTorch training epochs; returns per-epoch images/s and the measured window bounds."""
    out = WORK / "out" / f"{args.label}.weights"
    out.parent.mkdir(parents=True, exist_ok=True)
    for stale in (out, Path(str(out) + ".step")):
        stale.unlink(missing_ok=True)
    trainer = ([args.python, str(ROOT / "native/tools/pytorch_baseline.py"), "train"] if args.side == "pytorch"
               else [str(ROOT / args.exe)])
    cmd = trainer + [model["cfg"], "runs/ppm/train", "data/voc2007/yolo/labels/train", str(out),
           "--init", model["init"], "--init-layers", str(model["init_layers"]),
           "--epochs", str(args.epochs), "--limit", str(args.images), "--batch", str(args.batch),
           "--lr", "0.0001", "--burn-in", "1000", "--momentum", "0.9", "--seed", "7",
           "--flip-probability", "0.5", "--exposure-jitter", "0.3", "--saturation-jitter", "0.3",
           "--hue-jitter", "0.1", "--crop-jitter", "0.2", "--backend", "cuda"] + args.extra
    # Piped stdout is block-buffered, so epoch ends are timed by polling the .step sidecar the
    # trainer rewrites after every epoch checkpoint.
    marks, done = [], threading.Event()
    step = Path(str(out) + ".step")

    def watch():
        """Records the time of every change to the .step sidecar."""
        last = None
        while not done.is_set():
            try:
                text = step.read_text().strip()
            except OSError:
                text = None
            if text and text != last:
                marks.append(time.perf_counter())
                last = text
            time.sleep(0.02)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    rc = run_streaming(cmd, WORK / "logs" / f"{args.label}.log", lambda t, line: None)
    done.set()
    watcher.join()
    if rc != 0 or len(marks) < 2:
        raise SystemExit(f"native run failed rc={rc} epochs={len(marks)}")
    return [args.images / (b - a) for a, b in zip(marks, marks[1:])], marks


def darknet_cfg(src, dst, batch, max_batches):
    """Writes a Darknet cfg copy (LF endings) with fixed batch, subdivisions=1, max_batches, random=0."""
    out, section = [], None
    for line in Path(src).read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped
        key = stripped.split("=")[0].strip()
        if section == "[net]" and key in ("batch", "subdivisions", "max_batches"):
            continue
        if section in ("[yolo]", "[region]") and key == "random":
            continue
        out.append(line)
        if stripped == "[net]":
            out += [f"batch={batch}", "subdivisions=1", f"max_batches={max_batches}"]
        if stripped in ("[yolo]", "[region]"):
            out.append("random=0")
    Path(dst).write_bytes(("\n".join(out) + "\n").encode())


ITER = re.compile(r"^(\d+): [-\d.naif]+, [-\d.naif]+ avg, [\d.e-]+ rate, ([\d.]+) seconds, (\d+) images")


def darknet(args, model):
    """Times original Darknet training iterations in Docker; returns images/s and the window's bounds."""
    work = WORK / "darknet"
    work.mkdir(parents=True, exist_ok=True)
    cfg = work / f"{args.label}.cfg"
    darknet_cfg(ROOT / model["cfg"], cfg, args.batch, args.iters)
    (work / "voc.data").write_bytes(b"classes = 20\ntrain = /data/train.txt\nvalid = /data/valid.txt\n"
                                    b"names = /data/classes.txt\nbackup = /out\n")
    rel = lambda p: "/work/" + Path(p).resolve().relative_to(ROOT).as_posix()
    cmd = ["docker", "run", "--rm", "--gpus", "all",
           "--mount", f"type=bind,source={ROOT},target=/work",
           "--mount", f"type=bind,source={ROOT / 'data/voc2007/yolo'},target=/data",
           "--tmpfs", "/out",
           args.image, "stdbuf", "-oL", "-eL", "/opt/darknet/darknet", "detector", "train",
           rel(work / "voc.data"), rel(cfg), rel(ROOT / model["darknet_init"])]
    iters = {}

    def hook(t, line):
        """Records the arrival time of each iteration summary line."""
        m = ITER.match(line)
        if m:
            iters[int(m.group(1))] = t

    rc = run_streaming(cmd, WORK / "logs" / f"{args.label}.log", hook)
    lo, hi = args.window
    if rc != 0 or lo not in iters or hi not in iters:
        raise SystemExit(f"darknet run failed rc={rc} last={max(iters) if iters else None}")
    return [(hi - lo) * args.batch / (iters[hi] - iters[lo])], [iters[lo], iters[hi]]


def main():
    """Parses CLI options, runs one benchmark with telemetry and appends a JSON result line."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("side", choices=["native", "pytorch", "darknet"])
    p.add_argument("model", choices=sorted(MODELS))
    p.add_argument("--label", required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--images", type=int, default=1024, help="native/pytorch: images per epoch")
    p.add_argument("--epochs", type=int, default=4, help="native/pytorch: epochs (the first is warm-up)")
    p.add_argument("--exe", default=f"build/cuda_detector_train{EXE}")
    p.add_argument("--python", default=sys.executable, help="pytorch: interpreter with CUDA PyTorch")
    p.add_argument("--iters", type=int, default=200, help="darknet: max_batches")
    p.add_argument("--window", type=int, nargs=2, default=[20, 190], help="darknet: measured iterations")
    p.add_argument("--image", default="darknet-train:gpu")
    p.add_argument("--cool-to", type=int, default=0, help="wait until the GPU is at or below this many C")
    p.add_argument("--results", default=str(WORK / "training.jsonl"))
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra native/pytorch trainer flags")
    args = p.parse_args()
    (WORK / "logs").mkdir(parents=True, exist_ok=True)
    while args.cool_to and gpu_temperature() > args.cool_to:
        time.sleep(5)
    telemetry = Telemetry()
    try:
        rates, marks = darknet(args, MODELS[args.model]) if args.side == "darknet" else native(args, MODELS[args.model])
    finally:
        telemetry.stop()
    # Laptop clocks cannot be locked, so each window's mean SM clock is kept for clock-matched comparisons.
    window_mhz = [round(telemetry.window(a, b).get("sm_mhz_mean", 0.0)) for a, b in zip(marks, marks[1:])]
    record = {"label": args.label, "side": args.side, "model": args.model, "batch": args.batch, "extra": args.extra,
              "images_per_s": round(statistics.mean(rates), 2), "per_window": [round(r, 2) for r in rates],
              "per_window_sm_mhz": window_mhz, "gpu": telemetry.window(marks[0], marks[-1]),
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(record))
    with open(args.results, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    sys.exit(main())
