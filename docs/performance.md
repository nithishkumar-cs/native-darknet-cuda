# Performance against original Darknet and PyTorch

Measured 2026-11-02 against unmodified [pjreddie/darknet](https://github.com/pjreddie/darknet) at commit `f6afaab`, built with `GPU=1 CUDNN=0`. That is the original repository's only working GPU build on current CUDA: its cuDNN path calls `cudnnGetConvolutionForwardAlgorithm`, which cuDNN 8 removed. "cuBLAS" below is this framework's default build, which uses the same im2col + SGEMM convolution algorithm as Darknet's GPU path. "cuDNN" is the opt-in `-DDETECTOR_CUDNN=ON` build.

## Training throughput (laptop RTX 3070 Ti, 8 GB)

Both frameworks trained the same cfg on real VOC2007 images at a fixed 416x416 input, with the same per-step minibatch: 32 for YOLOv3-tiny, 16 for YOLOv2, and 8 for YOLOv3. Darknet used subdivisions 1 and `random=0`. Both applied crop, flip and HSV augmentation, and both started from pretrained backbones. Darknet ran in the `darknet-train:gpu` Docker image with 8 CPUs under WSL2; this framework ran natively on Windows. Darknet was GPU-bound (92-97% utilization), and its batch-loading wait was about 0.2 ms per iteration.

Each run started with the GPU at or below 60 C. Each measured roughly 60-80 s of steady state: whole epochs for this framework (the first epoch discarded), and an iteration window from 20 onward for Darknet. Checkpoint saves were included in both. The laptop cannot lock clocks, so the tables give mean SM clock alongside throughput. Rounds whose configurations ran at very different clocks (below 1,000 MHz under heat soak) are excluded from the speedups.

| Model, batch | Darknet img/s (SM MHz) | cuBLAS img/s (MHz) | cuDNN img/s (MHz) | cuBLAS vs Darknet | cuDNN vs Darknet |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv3-tiny, 32 | 129.7 (1153) | 159.3 (1181) | 330.8 (1300) | 1.23x | 2.55x |
| YOLOv2, 16 | 40.4 (1335) | 44.1 (1275) | 71.4 (1231) | 1.09x | 1.77x |
| YOLOv3, 8 (round 1) | 18.2 (1304) | 19.3 (1352) | 30.5 (1286) | 1.06x | 1.68x |
| YOLOv3, 8 (round 2) | 16.8 (1146) | 20.9 (1342) | 28.2 (1236) | 1.24x | 1.68x |

The cuDNN numbers above come from the build before the `cudnnFind*Ex` memory bound described below. The final build was re-measured in a third round. By then the laptop had been under load for hours and was heat-soaked. Darknet / cuDNN img/s (SM MHz) were 133.2 (1218) / 238.6 (677) for YOLOv3-tiny, 33.6 (1018) / 57.7 (878) for YOLOv2, and 13.6 (862) / 22.7 (838) for YOLOv3. That is 1.79x, 1.72x and 1.68x faster, even though the native build ran at the same or lower clocks in every pair.

To reproduce, run `native/tools/benchmark_training.py`. It writes one JSON line per run, including temperature, power, utilization and the mean SM clock of every measured window. For example:

```sh
python native/tools/benchmark_training.py darknet yolov3-tiny --label dn-tiny --batch 32 --iters 265 --window 20 260 --cool-to 60
python native/tools/benchmark_training.py native yolov3-tiny --label cudnn-tiny --batch 32 --images 4480 --epochs 6 --exe build-cudnn/cuda_detector_train.exe --cool-to 60
```

### Full training schedule

YOLOv3-tiny was trained over the full VOC2007 schedule: COCO-pretrained first 15 layers, about 320,000 images (71 epochs of 4,510, or 10,000 iterations of 32), LR 0.001 with burn-in 1,000 and steps at 8,000 and 9,000, at a fixed 416 input. Both runs were sustained, so the GPU spent most of each run thermally throttled. Both were evaluated with the same native evaluator: confidence 0.01, class-aware NMS IoU 0.45, and non-difficult objects.

| Run (laptop RTX 3070 Ti, 2026-11-02) | Wall time | Val mAP@0.5 | Test mAP@0.5 |
| --- | ---: | ---: | ---: |
| Original Darknet (batch 32, subdivisions 1) | 48.7 min | 0.58031 | 0.56575 |
| This framework, cuDNN build | **24.0 min** | 0.59529 | 0.57993 |

The Darknet time is its active training time. Windows idle-slept from 16:38:01 to 18:18:20 while the run was in its last iterations; the job had started at 15:49:54 and needed 36 s after wake to finish and copy its weights. A 2026-10-25 Darknet run of the same schedule, at subdivisions 16, took about 47 min and scored 0.59637 / 0.55069. The native run's final mean loss (9.4087) matches the earlier corrected native run (9.41512, 0.59801 / 0.58182).

### Training GPU memory

Peak `nvidia-smi` memory above idle while training, same batch and 416 input:

| Model, batch | Darknet | Native cuBLAS | Native cuDNN |
| --- | ---: | ---: | ---: |
| YOLOv3-tiny, 32 | 3,964 MiB | 3,354 MiB (-15%) | 4,016 MiB |
| YOLOv2, 16 | 5,470 MiB | 4,802 MiB (-12%) | 5,464 MiB |
| YOLOv3, 8 | 6,876 MiB | 5,948 MiB (-13%) | 6,590 MiB (-4%) |

The cuDNN figures include its one shared workspace, capped at 256 MiB. Algorithm benchmarking uses `cudnnFind*Ex` inside that workspace. The plain `cudnnFind*` calls allocate their own trial memory, and they briefly pushed YOLOv2/YOLOv3 training to about 7.9 GB on this 8 GB GPU.

### What made the difference

Nsight Systems traces drove each change:

1. **Overlapped data loading.** Before the change, images were decoded and augmented on the training thread, and GPU utilization was 30-44%. The next minibatch is now prepared on worker threads while the current one trains. Every random draw stays on the training thread, so results are identical for any thread count, and a regression test checks this.
2. **Hidden host-device transfers.** Inputs are staged in pinned memory and uploaded once per minibatch on a separate stream into double-buffered device inputs. `step()` no longer waits for its own backward pass, so the next upload overlaps it. The YOLO loss now runs on worker threads, with one head download and one upload per minibatch.
3. **One launch per layer, not per sample.** Batch norm, activations, maxpool, upsample, reorg, shortcut and route kernels each run once over the whole minibatch. Pointwise convolutions skip im2col. Head convolutions use SGEMM instead of a direct kernel. Per-thread BN partial sums use fp32, because GeForce fp64 runs at 1/64 rate.
4. **Benchmarked cuDNN algorithms.** `cudnnFind*Ex` measures the fastest full-FP32 forward and backward algorithms on the real tensors for each shape and batch size, within a workspace cap. All layers share one device workspace. The previous build hard-coded `IMPLICIT_GEMM` and `ALGO_0`.
5. **Less activation memory.** No pre-activation copy is stored, because leaky ReLU keeps the sign and backward can read it from the activated output. YOLOv3 batch-8 training fell from about 7.9 GB to 6.2 GB of GPU memory. Before this, it was spilling into shared system memory on the 8 GB card.

## Training against PyTorch (laptop RTX 3070 Ti)

Measured 2026-11-03 against PyTorch 2.11.0 (CUDA 12.8, cuDNN 9.19) with `native/tools/pytorch_baseline.py`, a PyTorch version of the same training job written for this comparison. It builds the network from the same Darknet `.cfg`, loads and saves Darknet `.weights`, and copies this framework's YOLO/region loss, crop/flip/HSV augmentation, SGD with momentum and weight decay, learning-rate schedule and batch-norm settings. Its command line matches `cuda_detector_train`. It uses the usual PyTorch speed settings: `cudnn.benchmark` autotuning, and eight DataLoader worker processes with pinned memory that send uint8 images (about 740 img/s on their own, over twice what any run used). Loss targets are built in the workers, so a training step never waits on the host. `torch.compile` is unavailable because this Windows install has no Triton, and `channels_last` was about 4x slower, so neither is used.

The two implementations were checked against each other before timing:

- **Forward pass.** On the three trained checkpoints, PyTorch's head outputs match ONNX Runtime on the exported graph within 3e-5, for logits up to 30. ONNX Runtime matches native inference exactly.
- **One training step.** Both started from the same checkpoint and trained on the same first 32/16/8 images, with no augmentation (`--epochs 1 --lr 1 --burn-in 0`, plus `--exact-inputs` for PyTorch). The losses agree to five significant figures: 12.466 vs 12.4659 for YOLOv3-tiny, 14.7505 for both on YOLOv2, and 6.5689 vs 6.5686 for YOLOv3.
- **Parameter updates.** Every weight, bias and BN-scale update is within 0.33% of the native cuDNN build's. Native's own cuBLAS and cuDNN builds differ from each other by up to 0.40%. PyTorch keeps an unbiased variance in its running statistics, so running-variance updates differ by up to 1.7%.
- **Regression test.** `native/tests/test_pytorch_baseline.py` re-checks the forward pass, the weight files and closed-form loss gradients.

On this day, the laptop throttled within 30-40 s of sustained load, and its idle temperature was 66-69 C. Each run therefore started after 180 s idle at or below 70 C, discarded a warm-up epoch, and measured two ~10 s epochs of 3,200, 800 or 320 images. Native and PyTorch FP32 each ran twice per model, in A-B-B-A order. The table gives mean img/s, with the mean SM clock in MHz.

| Model, batch | Native cuDNN, FP32 | PyTorch FP32 | PyTorch TF32 (its default) | PyTorch AMP (FP16) |
| --- | ---: | ---: | ---: | ---: |
| YOLOv3-tiny, 32 | **333.7** (1346) | 295.6 (1367) | 330.3 (1446) | 420.5 (1530) |
| YOLOv2, 16 | **70.1** (1254) | 55.7 (1018) | 85.8 (1521) | 119.9 (1474) |
| YOLOv3, 8 | **27.6** (1148) | 27.1 (1070) | 39.4 (1483) | 60.8 (1594) |

At the same FP32 precision, native training is 1.13x faster than PyTorch for YOLOv3-tiny, 1.26x for YOLOv2 and level (1.02x) for YOLOv3. PyTorch's FP32 YOLOv2 runs settled at lower clocks. Per MHz, the YOLOv2 and YOLOv3 pairs are within 5% of each other, and YOLOv3-tiny stays 1.15x in native's favour. The longer 60-80 s windows used for Darknet above heat-soaked most runs to 600-1,000 MHz on this day. The pairs that held matched clocks agree: native vs PyTorch FP32 was 307.5 (1225) vs 287.1 (1202) for YOLOv3-tiny and 70.4 (1227) vs 65.0 (1190) for YOLOv2.

This framework computes in strict FP32 only, using cuDNN FMA math and pedantic cuBLAS. PyTorch's default TF32 tensor-core convolutions match native speed on YOLOv3-tiny and are 1.2x and 1.4x faster on YOLOv2 and YOLOv3. With mixed precision, PyTorch is 1.3x, 1.7x and 2.2x faster.

### Full schedule against PyTorch

The full YOLOv3-tiny schedule above was repeated in PyTorch FP32 on 2026-11-03. It alternated with native cuDNN in two pairs, native first in pair 1 and PyTorch first in pair 2. Each run started after 180 s idle at or below 70 C, and the native evaluator scored every checkpoint.

| Run (2026-11-03) | Wall time | Val mAP@0.5 | Test mAP@0.5 | Final mean loss |
| --- | ---: | ---: | ---: | ---: |
| Native cuDNN, pair 1 | 26.1 min | 0.59119 | 0.57958 | 9.4125 |
| PyTorch FP32, pair 1 | 23.5 min | 0.59565 | 0.57754 | 9.5146 |
| PyTorch FP32, pair 2 | 23.3 min | 0.60590 | 0.57681 | 9.5318 |
| Native cuDNN, pair 2 | 31.1 min | 0.58404 | 0.58053 | 9.4332 |

Accuracy matches within the spread between runs. Wall time does not follow the short-run results. Over 23-31 minutes of continuous training, the GPU sat at its 88 C thermal limit (logged for pair 2), and the native runs throttled harder:

- In pair 2, native averaged 323 MHz at 72 W, against 901 MHz at 96 W for PyTorch.
- Native epochs slowed from 13 s to 47 s, while PyTorch's mostly held at 18-20 s.
- In short runs of the same job, the native trainer kept about 5.5 CPU cores busy, against PyTorch's 3.9. The laptop's CPU and GPU share cooling.

Why native throttles harder has not been isolated. The same native run took 24.0 min on 2026-11-02.

```sh
python native/tests/test_pytorch_baseline.py   # needs torch, onnx, onnxruntime (CPU is enough)
python native/tools/benchmark_training.py pytorch yolov3-tiny --label pt-tiny --batch 32 --images 3200 --epochs 3 \
    --python <python with CUDA PyTorch> --cool-to 70 --extra --precision fp32
```

## Inference latency (Jetson Orin Nano Super, 25 W, JetPack R36.5)

Both frameworks were timed with the same protocol on held-out VOC2007 test image `002570.jpg`. The input was letterboxed once. After 10 untimed warmup calls, 100 timed calls each covered input upload, the full forward pass, and the detection-head download. For Darknet, `native/tools/darknet_bench.c` times `network_predict` through Darknet's public API, built against an unmodified `libdarknet.a`. For this framework, the measurement is `cuda_detector benchmark --scope network`. Each value is the mean in ms. Darknet was measured in three separate sessions, and those agreed within 0.6 ms. The cuDNN build was measured twice per model, and the table gives the average.

| Model | Darknet GPU | Native cuBLAS | Native cuDNN | TensorRT FP32 | TensorRT FP16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv3-tiny | 15.1 | 13.4 | 8.8 | 4.3 | 3.8 |
| YOLOv2 | 55.7 | 51.8 | 44.1 | 12.5 | 6.8 |
| YOLOv3 (106 layers) | 116.3 | 100.3 | 75.0 | 22.6 | 9.9 |

Against Darknet, native cuDNN is 1.7x, 1.3x and 1.6x faster, and TensorRT FP16 is 4.0x, 8.2x and 11.7x faster. The previous native CUDA convolution was a direct per-output-pixel kernel that ran at 91 ms, 578 ms and 1,246 ms, so the im2col + SGEMM rewrite with a fused BN/bias/leaky epilogue is 7-12x faster than it. Validation mAP@0.5 on Jetson was unchanged after the rewrite: 0.60006 for YOLOv3-tiny and 0.62139 for YOLOv3, in both builds.

On the laptop RTX 3070 Ti, the same protocol gave the following means over two rounds. Darknet ran in Docker under WSL2; this framework ran natively on Windows.

| Model | Darknet GPU | Native cuBLAS | Native cuDNN |
| --- | ---: | ---: | ---: |
| YOLOv3-tiny | 3.01 | 2.34 | 2.10 |
| YOLOv2 | 8.71 | 7.61 | 8.70 |
| YOLOv3 | 18.80 | 15.86 | 14.43 |

PyTorch 2.11 on the same laptop on 2026-11-03 used the same protocol, timed by `pytorch_baseline.py bench-infer`. The table gives means over two rounds, with native measured in the same session. PyTorch runs the plain eval-mode model, with batch norm not folded into the convolutions. FP16 means a half-precision model and input.

| Model | Native cuBLAS | Native cuDNN | PyTorch FP32 | PyTorch TF32 | PyTorch FP16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv3-tiny | 2.26 | **2.10** | 2.25 | 2.32 | 1.92 |
| YOLOv2 | **7.52** | 8.46 | 8.61 | 6.54 | 5.03 |
| YOLOv3 | 15.32 | **13.52** | 15.09 | 9.90 | 7.25 |

The faster native build is 1.07x, 1.15x and 1.12x faster than PyTorch FP32. PyTorch's TF32 and FP16 tensor-core paths are faster still, except TF32 on YOLOv3-tiny.

Reproduce with the [native detector commands](native-detector.md). For Darknet:

```sh
gcc -O2 -DGPU -Idarknet/include -I/usr/local/cuda/include native/tools/darknet_bench.c darknet/libdarknet.a \
    -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lstdc++ -lm -pthread -o darknet_bench
./darknet_bench yolov3-tiny-voc-b1.cfg model.weights 002570.jpg 10 100
```

The `-b1` cfg copies set `batch=1` and `subdivisions=1`. Weights trained here load unchanged in Darknet. Darknet's YOLOv2 reorg channel order differs from this framework's, so its YOLOv2 detections differ, but the amount of work is the same.
