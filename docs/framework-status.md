# Framework status

This project is an independent C++/CUDA neural-network framework inspired by [pjreddie/darknet](https://github.com/pjreddie/darknet). It keeps the original CUDA CNN in Git history and implements configurable training and inference in `native/`. Supported Darknet cfg layers and weight files are compatible where documented; unsupported layers fail explicitly.

## Working paths

- Config-driven CPU/CUDA graph inference and training for convolution, max-pooling, global `[avgpool]`, connected, route, shortcut, upsample, reorg, dropout, sample-wide `[l2norm]`, channel-local `[normalization]`, spatially unshared `[local]`, transposed `[deconvolutional]`, YOLO and region heads, and terminal softmax. Batch normalization is trainable with true minibatch statistics in the detector trainers. Classifier examples cover linear, MLP and CNN models.
- RNN, LSTM and GRU cfg layers have persistent sequence state, CPU/CUDA inference, and truncated backpropagation through time in the CPU/CUDA classifier trainers. The detector trainers reject recurrent layers with a clear error.
- Native CUDA detector training includes anchor assignment and YOLO/region loss, crop/flip/HSV augmentation, multiscale resizing, burn-in/step-decay scheduling, SGD momentum and weight decay. Loader threads prepare the next minibatch, and pinned, double-buffered uploads overlap the previous backward pass. The default convolution path is im2col + cuBLAS SGEMM for training and inference. `-DDETECTOR_CUDNN=ON` adds cuDNN convolution, with per-shape benchmarked algorithms, and spatial batch normalization. On the same GPU, the cuDNN build trains 1.7-2.6x faster than original Darknet; [performance.md](performance.md) has the method and numbers.
- CUDA detector training can use multiple selected devices via `--gpus 0,1`: one replica per device, balanced shards of the total `--batch`, concurrent local steps, and sample-weighted state synchronization after each global step. The one-GPU path and weighted synchronization math are tested. This laptop has only one GPU, so an actual two-GPU run and its throughput have not been measured.
- Darknet-format weight load/save, partial backbone initialization, checkpoint step/epoch resume, IDX/MNIST classifier CLI, native detection/evaluation, and Jetson inference. Optional ONNX/TensorRT deployment has been validated for YOLOv3-tiny, YOLOv2/Darknet19, and full YOLOv3/Darknet-53.
- Semantic segmentation now has a pixelwise softmax-cross-entropy head with a void-label ignore value; instance segmentation uses a separately trained binary ROI mask head on each detector box. VOC2007 semantic/object masks can be packed into fixed-size records. Synthetic training/evaluation, CPU/CUDA mask inference and CPU/CUDA training parity pass. Real VOC2007 runs (209 train / 213 val images, 2026-10-29): semantic 64x64 encoder-decoder 0.302 train / 0.060 val mean IoU (overfits the tiny split); mask head 0.468 val GT-crop mask IoU; no end-to-end instance AP.

## Measured model results

All detector scores below use the same native CUDA evaluator with confidence 0.01, class-aware NMS IoU 0.45 and all-point AP@0.5. VOC2007 labels were converted without difficult objects; these are not official VOC leaderboard scores.

| Checkpoint origin and architecture | Validation: 501 images, 1,305 objects | Test: 4,952 images, 12,032 objects |
| --- | ---: | ---: |
| This framework, YOLOv3-tiny with COCO backbone initialization | 0.59801 | 0.58182 |
| This framework, YOLOv2 / Darknet19 with pretrained backbone | 0.56054 | 0.57215 |
| This framework, full YOLOv3 / Darknet-53 with pretrained backbone | 0.62098 | 0.61593 |
| Historical original-Darknet YOLOv3-tiny weights, evaluated here | 0.57612 | 0.55534 |

All three native detector architectures were also evaluated on Jetson with native CUDA and TensorRT FP32/FP16. For YOLOv2, held-out test mAP@0.5 was 0.57177/0.57176/0.57151 across those backends; for full YOLOv3 it was 0.61502/0.61479/0.61517. [Engineering milestones](engineering-log.md) record the timing scope and values. Historical full-dataset MNIST Jetson runs of the project's classifier trainer scored 97.24% test accuracy for an MLP and 97.27% for a small CNN.

A deeper two-convolution MNIST classifier trained by this framework on the laptop RTX 3070 Ti for three epochs (60,000 training images per epoch, seed 2026, LR 0.01) scored **98.49%** on all 10,000 held-out test images. CPU and CUDA inference agreed on the correct count (9,849). Its checkpoint and logs remain local ignored artifacts.

## Current limits

- This is the supported subset of Darknet's cfg vocabulary, not an arbitrary upstream cfg interpreter. A same-evaluator external-Darknet YOLOv2 baseline has not been run.
- Actual multi-GPU execution needs a host with at least two CUDA devices. Each device uses local batch-normalization statistics during its forward/backward pass; running statistics are averaged between devices. The current cross-device state transfer uses host memory, and its throughput is unmeasured.
- Native training from scratch has not produced a useful VOC detector; the measured native checkpoints above use pretrained backbones.
- The segmentation example trainers run on CPU (`VectorTrainer`) or GPU (`--backend cuda`, host-computed loss with CUDA layer kernels, single-sample SGD; evaluation is CPU). The instance mask stage is class-agnostic and trained separately from the detector; its `eval` metric is mask IoU on supplied ground-truth crops, not end-to-end instance AP. Mask inference supports native CUDA, but ONNX/TensorRT export and Jetson validation of these new heads have not been done.

See [engineering milestones](engineering-log.md) and [the training design](../DESIGN.yaml) for the implementation and validation history.
