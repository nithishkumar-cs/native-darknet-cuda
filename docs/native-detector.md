# Native detector: scope and measured status

**Historical snapshot:** this page records the early YOLOv3-tiny inference and CPU pilot stage. Its statements about unsupported full YOLOv3, missing native CUDA detector training and no useful native detector are superseded by [current framework status](framework-status.md) and the [engineering log](engineering-log.md). The original measurements below remain for provenance.

This repository builds an independent C++/CUDA detector inspired by [pjreddie/darknet](https://github.com/pjreddie/darknet), without copying or forking its source. The original MNIST CUDA CNN remains in the source tree and Git history. Earlier unrelated PyTorch/Ultralytics work was reversed locally without rewriting history; ignored local `data/` and `runs/` were retained. No upstream `.cfg`, pretrained `.weights`, or dataset files are committed.

## What works

- Parse the layers required by the [original YOLOv3-tiny configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg): convolution, maxpool, route, upsample, and YOLO heads. Unsupported layers fail explicitly. Full YOLOv3 also needs shortcut layers and is not supported.
- Read matching Darknet binary weights, including batch-normalization parameters; reject truncated or trailing data.
- Run CPU reference or native CUDA inference, decode two YOLO heads, apply class-aware NMS, and map boxes through letterboxing. A CUDA session keeps weights and intermediate device buffers resident between inferences.
- Read PPM input, or JPEG/PNG when built with OpenCV. The CLI provides inspection, detection, network-only and full-pipeline timing, experimental CPU training, and class-wise AP@0.5 evaluation.
- Train a small Darknet-style detector using single-image CPU SGD and a simplified YOLO objective, then save a Darknet-format checkpoint. The included `native/tests/voc-smoke.cfg` uses 64x64 input with two heads and 20 VOC classes; it is **not** YOLOv3-tiny.

The experimental trainer is a research baseline, not Darknet training parity. It uses fixed batch-normalization statistics/scales, batch size 1, a simplified BCE/MSE detection objective, and CPU backpropagation. It has no native CUDA backprop, full Darknet target/loss behavior, augmentation pipeline, scheduled learning rate, or demonstrated useful VOC accuracy. The convolution CUDA kernel is a simple implementation, not cuDNN or TensorRT.

## Build and commands

CPU build (CMake 3.18+, C++17 compiler):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDETECTOR_CUDA=OFF -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Jetson Orin CUDA build with OpenCV development libraries:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDETECTOR_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DDETECTOR_OPENCV=ON -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

Set `CMAKE_CUDA_ARCHITECTURES` for the actual GPU. Without `DETECTOR_OPENCV=ON`, use PPM images. On Windows, the local CUDA test used Ninja with an initialized MSVC environment because the Visual Studio generator did not find a CUDA toolset. Three C++ and three standard-library Python CTest suites cover native inference/training/metrics and the dataset/training helpers; all six passed on the Jetson CUDA/TensorRT/OpenCV build. A seventh ONNX export test is registered when `numpy`, `onnx`, and `onnxruntime` are installed; all seven passed on the Windows CPU build.

```sh
./build/cuda_detector inspect /path/to/yolov3-tiny.cfg
./build/cuda_detector detect /path/to/yolov3-tiny.cfg /path/to/yolov3-tiny.weights image.jpg --backend cuda --threshold 0.25
./build/cuda_detector benchmark /path/to/yolov3-tiny.cfg /path/to/yolov3-tiny.weights image.jpg --backend cuda --scope network --warmup 5 --iterations 30
./build/cuda_detector benchmark /path/to/yolov3-tiny.cfg /path/to/yolov3-tiny.weights image.jpg --backend cuda --scope pipeline --warmup 5 --iterations 30
./build/cuda_detector_train native/tests/voc-smoke.cfg images/train labels/train detector.weights --epochs 1 --lr 0.001 --seed 7
./build/cuda_detector_eval native/tests/voc-smoke.cfg detector.weights images/val labels/val --backend cuda --confidence 0.01
```

The experimental trainer expects flat image directories and matching YOLO text-label directories (`class cx cy width height`, normalized to the source image). It refuses to overwrite an existing output checkpoint. `--init matching.weights`, `--limit N`, `--decay VALUE`, and multiple `--epochs` are available. Never pair COCO YOLOv3-tiny weights with the 20-class smoke configuration.

## Jetson inference validation

> The native CUDA timings in this section predate the 2026-11-02 rewrite. They came from a
> direct per-output-pixel convolution kernel; the current im2col + cuBLAS SGEMM path runs this
> network in 13.4 ms, or 8.8 ms with cuDNN. The same-protocol Darknet comparison is in
> [performance.md](performance.md). Darknet's `network_predict` timer, cited below, measured
> cold single runs, not warm iterations.

The board was an NVIDIA Jetson Orin, JetPack R36.5.0, CUDA 12.6.68, OpenCV 4.8, in its existing 25W power mode. The native detector was built in Release for CUDA architecture 87 with OpenCV. With the unmodified upstream YOLOv3-tiny configuration and official matching COCO weights, native CPU and CUDA produced one `person` detection at score 0.4172 on a held-out VOC2007 test image; original Darknet CPU/GPU reported `person: 40%`. This is approximate visible-detection agreement, **not** layer-by-layer or raw-tensor parity proof.

Native CUDA timing with that pretrained COCO model, five warmups and 30 measured iterations at 416x416:

| Scope | Mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| Network only | 94.975 ms | 94.962 ms | 95.115 ms |
| In-memory pipeline | 104.749 ms | 104.880 ms | 104.923 ms |

Both exclude image-file I/O, configuration/weight loading, and CUDA session setup. The original Darknet GPU build's own `network_predict` timer averaged 124.543 ms over five separate runs on the same board/image/configuration. It uses a different timing implementation, so this is context, **not** a rigorous speedup claim. Native CPU network-only time was about 8.52 seconds because its convolution is an untuned reference implementation.

## Actual YOLOv3-tiny fine-tuning

The [VOC2007 training workflow](yolov3-tiny-training.md) used the original Darknet GPU trainer externally to finish 10,000 batches of the actual YOLOv3-tiny model. The native CUDA runtime evaluated every validation checkpoint and the selected final model. Validation mAP@0.5 rose from 0.00372 at batch 400 to **0.57379 at batch 10,000** on 501 images; that validation-selected model reached **0.55534 on all 4,952 held-out test images**. The AP calculation uses converted non-difficult YOLO labels and is not an official VOC or COCO benchmark number. The final model gave a `person` score of 0.9967 on a held-out test image in the Windows native CUDA build.

Final trained-model Jetson CUDA timing, same image/procedure and existing 25W mode: network-only mean **91.204 ms** (p50 91.154, p95 91.428) and in-memory pipeline mean **100.786 ms** (p50 100.918, p95 101.076). Different weights can change detections and postprocessing work; these are separate measurements, not a speedup claim against the pretrained run or original Darknet.

The **separate experimental native CPU trainer** completed one 64x64 epoch over 4,510 VOC2007 train images and yielded mAP@0.5 of only 0.00113 on 501 validation images at confidence 0.01. It produced 1,479,209 detections and is not a useful model. The SSH connection dropped after its checkpoint was saved, so its epoch loss and elapsed training time were not retained and are not claimed. This pilot did **not** train YOLOv3-tiny.

## ONNX and TensorRT deployment

The [optional ONNX/TensorRT workflow](onnx-tensorrt.md) exports the **same** trained Darknet YOLOv3-tiny weights directly to ONNX and runs Jetson-built FP32/FP16 engines through the existing native decoder and evaluator. On all 4,952 held-out test images, TensorRT FP32 scored 0.55598 mAP@0.5 and FP16 scored 0.55595, versus 0.55534 for native CUDA. On the same 416x416 test image and five-warmup/30-run procedure, mean in-memory pipeline time was 20.008 ms for FP32 and 11.970 ms for FP16. These engines are optional deployment artifacts; the native Darknet-format path still works without either dependency.

## Next technical work

For a fully self-contained Darknet-equivalent **training** project, implement and verify faithful YOLO target assignment/loss, batch normalization and backprop, augmentation, and learning-rate scheduling in the native trainer, then retrain and reevaluate. For stronger inference claims, compare raw layer/head tensors against original Darknet and ONNX/TensorRT within numerical tolerances and profile/tune the native CUDA convolution. Full YOLOv3 is an optional larger-model comparison, not a requirement for this deployment.
