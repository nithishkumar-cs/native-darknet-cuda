# native-darknet-cuda

An independently written C++/CUDA neural-network framework inspired by [pjreddie/darknet](https://github.com/pjreddie/darknet). It is not a fork and does not contain Darknet source. `native/` contains the configurable framework; the original fixed-shape MNIST CUDA CNN remains in the repository. Training runs on the laptop GPU and Jetson is an inference target. See [current support and limits](docs/framework-status.md).

The graph supports convolution, max-pooling, global average pooling, dense (`[connected]`), route, shortcut, upsample, reorg, dropout, sample-wide L2 normalization, YOLO and region detector heads, softmax, and RNN/LSTM/GRU layers. Classifiers and detectors have CPU and CUDA training paths; recurrent classifiers support sequence inference and truncated backpropagation through time on both. Detector training uses true minibatches, trainable batch normalization, augmentation, a learning-rate schedule, and SGD momentum. Optional cuDNN kernels can replace the default cuBLAS convolution path. Unsupported cfg layers fail explicitly.

Three detector architectures have been trained end to end by this framework on VOC2007 with pretrained backbones. Using the same native CUDA evaluator at confidence 0.01 and class-aware NMS IoU 0.45:

| Native-trained architecture | Validation mAP@0.5, 501 images | Test mAP@0.5, 4,952 images |
| --- | ---: | ---: |
| YOLOv3-tiny | 0.59801 | 0.58182 |
| YOLOv2 / Darknet19 | 0.56054 | 0.57215 |
| Full YOLOv3 / Darknet-53 | 0.62098 | 0.61593 |

These are this project's all-point AP calculations on converted VOC labels excluding difficult objects, not official VOC leaderboard scores. The earlier **external-Darknet** YOLOv3-tiny checkpoint remains a historical comparison (0.57612 validation / 0.55534 test with the comparable native evaluator); it is not evidence of native training. Checkpoints, datasets and run logs are local ignored artifacts. The corrected native YOLOv3-tiny checkpoint was evaluated on Jetson with native CUDA and optional TensorRT FP32/FP16; see [Jetson results](docs/onnx-tensorrt.md).

Against unmodified original Darknet on the same hardware (see [performance.md](docs/performance.md)), the cuDNN build trains YOLOv3-tiny / YOLOv2 / YOLOv3 2.55x / 1.77x / 1.68x faster on an RTX 3070 Ti laptop GPU. It finished the full YOLOv3-tiny VOC schedule in 24.0 min, against 48.7 min for Darknet, at 0.580 versus 0.566 test mAP@0.5. On a Jetson Orin Nano, native CUDA inference is 1.3-1.7x faster than Darknet, and the TensorRT FP16 export path is 4.0-11.7x faster.

Build on Jetson Orin with CUDA and OpenCV development libraries:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDETECTOR_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 -DDETECTOR_OPENCV=ON -DBUILD_TESTING=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

For CPU-only builds, use `-DDETECTOR_CUDA=OFF` and omit `CMAKE_CUDA_ARCHITECTURES`. Add `-DDETECTOR_CUDNN=ON` to an existing CUDA build to use the optional cuDNN kernels after staging the SDK with `native/tools/setup_cudnn.ps1`. Without OpenCV, input images must be PPM. Select the CUDA architecture for your GPU.

Train future models on the laptop GPU. The [deeper CNN example](native/examples/mnist-deep-conv.cfg), trained here for three epochs on all 60,000 MNIST images, reached **98.49% test accuracy** with matching CPU/CUDA evaluation. Historical [full-dataset MNIST Jetson runs](docs/mnist-jetson.md) reached **97.24%** for a two-layer dense model and **97.27%** for a small CNN. Use a fresh output checkpoint name because the CLI refuses to overwrite existing weights:

```sh
./build/cuda_classifier train native/examples/mnist-deep-conv.cfg data/mnist/train-images-idx3-ubyte data/mnist/train-labels-idx1-ubyte runs/mnist/new-deep-conv.weights --backend cuda --epochs 3 --lr 0.01
./build/cuda_classifier eval native/examples/mnist-deep-conv.cfg data/mnist/t10k-images-idx3-ubyte data/mnist/t10k-labels-idx1-ubyte runs/mnist/new-deep-conv.weights --backend cuda
```

```sh
./build/cuda_detector inspect runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg
./build/cuda_detector detect runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights image.jpg --backend cuda --names runs/yolov3-tiny-voc/voc.names
./build/cuda_detector benchmark runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights image.jpg --backend cuda --scope pipeline --warmup 5 --iterations 30
./build/cuda_detector_eval runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg runs/yolov3-tiny-voc/final.weights images/test labels/test --backend cuda --confidence 0.01
```

For TensorRT, export with `native/tools/export_darknet_onnx.py`, build the engine on the Jetson, and configure with `-DDETECTOR_TENSORRT=ON`; see the [tested commands and results](docs/onnx-tensorrt.md). Full YOLOv3 is not required for this path.

Detector training accepts `--backend cuda --gpus 0,1 --batch N` on a host with two GPUs; `N` is the total minibatch size. The default is GPU 0. The code synchronizes learned weights, momentum and batch-normalization running state after each global step. This laptop has one RTX 3070 Ti, so two-device execution and throughput remain unmeasured. Its one-GPU path and synchronization math have CUDA regression tests.

## Segmentation examples

`cuda_segmenter` trains a semantic head with pixelwise softmax cross-entropy and reports pixel accuracy and mean class IoU. `cuda_instance_segmenter` trains a class-agnostic binary mask head from object crops, then pairs that head with any supported detector at inference: one separate mask is predicted per detected box and written as a full-image PGM with an `instances.tsv` index. Both heads train on the CPU (`VectorTrainer`, default) or on the GPU (`--backend cuda`, `CudaClassificationTrainer::step_vector`, CPU/CUDA parity-tested for mse, sigmoid-BCE and pixel-softmax losses); mask inference supports CPU and native CUDA. Measured on this laptop (VOC2007 segmentation split, only 209 train / 213 val images): `native/examples/segmenter-deep.cfg` (64x64 encoder-decoder, 400 epochs, lr 0.05, ~4 min on the RTX 3070 Ti) reaches 0.302 train but only 0.060 val mean IoU (0.706 val pixel accuracy, i.e. heavy overfit; the tiny split is the limit), and the class-agnostic mask head (60 epochs) reaches 0.482 train / 0.468 val ground-truth-crop mask IoU. ONNX export (`native/tools/export_darknet_onnx.py`) now also handles head-less segmentation cfgs (`[deconvolutional]` -> ConvTranspose; output `logits`, raw per-pixel logits). ORT on the exported val-set models reproduces the native CPU `eval` numbers exactly (semantic 0.706361 pixel accuracy / 0.059803 mIoU; mask 0.468370 IoU). On the Jetson Orin Nano (trtexec, batch 1, GPU compute mean): semantic 64x64 0.173 ms FP32 / 0.125 ms FP16, mask head 32x32 0.052 ms FP32 / 0.052 ms FP16; no native TensorRT session wrapper for these heads. The instance stage is trained separately from the detector and its `eval` command measures mask IoU on ground-truth crops, not end-to-end instance AP.

For VOC2007 segmentation data already present locally, generate manifests and fixed-size float32 records (Pillow and NumPy required for packing):

```sh
python native/tools/prepare_voc_segmentation.py data/voc2007/VOCdevkit/VOC2007 train runs/voc-seg
python native/tools/pack_segmentation.py semantic runs/voc-seg/train-semantic.tsv runs/voc-seg/train-semantic-32.vecbin --width 32 --height 32 --classes 21 --ignore-value 255
python native/tools/pack_segmentation.py instance runs/voc-seg/train-instance.tsv runs/voc-seg/train-instance-32.vecbin --width 32 --height 32
./build/cuda_segmenter train native/examples/segmenter.cfg runs/voc-seg/semantic.weights runs/voc-seg/train-semantic-32.vecbin --epochs 1 --lr 0.01
./build/cuda_instance_segmenter train native/examples/instance-mask.cfg runs/voc-seg/mask.weights runs/voc-seg/train-instance-32.vecbin --epochs 1 --lr 0.01
```

The manifests use tab-separated image/mask paths; instance rows also hold integer box bounds and the object's palette ID. Semantic records append one class ID per output pixel (`-1` ignores a void pixel); instance records append a binary mask. The 32×32 cfgs are small examples, not tuned VOC models. Use the separate `val` split and `eval` commands for a real held-out measurement. Prediction takes detector cfg/weights, mask cfg/weights, an image and an empty output directory; run `cuda_instance_segmenter` with no arguments for its exact command syntax.

Do not pair 80-class COCO weights with the 20-class VOC config. The included `native/tests/voc-smoke.cfg` is **not** YOLOv3-tiny. See [current framework status](docs/framework-status.md) for the implementation's limits.
