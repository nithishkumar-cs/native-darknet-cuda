# Performance against original Darknet

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

To reproduce, run `native/tools/benchmark_training.py`. It writes one JSON line per run, including temperature, power and utilization. For example:

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

Reproduce with the [native detector commands](native-detector.md). For Darknet:

```sh
gcc -O2 -DGPU -Idarknet/include -I/usr/local/cuda/include native/tools/darknet_bench.c darknet/libdarknet.a \
    -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lstdc++ -lm -pthread -o darknet_bench
./darknet_bench yolov3-tiny-voc-b1.cfg model.weights 002570.jpg 10 100
```

The `-b1` cfg copies set `batch=1` and `subdivisions=1`. Weights trained here load unchanged in Darknet. Darknet's YOLOv2 reorg channel order differs from this framework's, so its YOLOv2 detections differ, but the amount of work is the same.
