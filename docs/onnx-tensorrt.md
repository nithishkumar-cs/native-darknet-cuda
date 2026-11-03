# Darknet YOLOv3-tiny to ONNX and TensorRT

This optional path deploys the project's **native-trained YOLOv3-tiny checkpoint**. The later section preserves measurements from the historical checkpoint trained with original Darknet. It does not train a new architecture or use Ultralytics/PyTorch. `native/tools/export_darknet_onnx.py` reads the 20-class Darknet `.cfg` and `.weights` directly and writes standard ONNX operators. No Darknet source is copied into this project. The input `.cfg` is an adaptation of [pjreddie/darknet's YOLOv3-tiny config](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg); retain its attribution header and consult the upstream [LICENSE](https://github.com/pjreddie/darknet/blob/master/LICENSE) and [MIT notice](https://github.com/pjreddie/darknet/blob/master/LICENSE.mit) before redistributing it.

The exported ONNX graph has fixed input `images: float32[1,3,416,416]` and two **raw**, unactivated NCHW YOLO logits outputs, `head_0: [1,75,13,13]` and `head_1: [1,75,26,26]`. It folds Darknet batch normalization into convolutions and uses standard Conv, LeakyRelu, MaxPool, Identity, Resize, and Concat nodes. Letterboxing, sigmoid/box decoding, confidence filtering, class-aware NMS, and inverse letterboxing remain outside the graph. The optional TensorRT backend returns these same raw heads to the existing native C++ decoder/evaluator; no custom YOLO plugin is needed.

## Selected native-trained checkpoint (2026-10-12)

The laptop-trained `runs/yolov3-tiny-voc-pretrained-full-budget-corrected.weights` (SHA-256 `25acc5731ca6ba53b25db0350a0564550acad2661d6bef78e17b969df6809841`) is the selected Jetson deployment checkpoint. Export with the same 20-class cfg:

```sh
python3 native/tools/export_darknet_onnx.py \
  --cfg runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg \
  --weights runs/yolov3-tiny-voc-pretrained-full-budget-corrected.weights \
  --output runs/yolov3-tiny-voc-pretrained-full-budget-corrected.onnx
```

The evaluated ONNX file had SHA-256 `692ba3711c1b92c49266ee574ee936fb045891f61c04ec0c34d9af9e78636d24`. TensorRT engines were built on the Jetson from that ONNX model, then all three backends were evaluated with confidence 0.01 and class-aware NMS IoU 0.45. Validation contained 501 images and 1,305 non-difficult objects; test contained 4,952 images and 12,032 such objects.

| Backend | Validation mAP@0.5 | Test mAP@0.5 | Network mean | In-memory pipeline mean |
| --- | ---: | ---: | ---: | ---: |
| Native CUDA | 0.60006 | 0.58119 | 91.138 ms | 100.346 ms |
| TensorRT FP32 | 0.60016 | 0.58103 | 6.386 ms | 19.083 ms |
| TensorRT FP16 | 0.59888 | 0.58066 | 3.742 ms | 12.216 ms |

Timing used five warmups and 30 measured iterations on the same test image in the board's existing power mode. The 2026-11-02 native CUDA convolution rewrite cut the native network mean from 91.138 ms to 13.4 ms (cuBLAS) or 8.8 ms (cuDNN), with validation mAP unchanged at 0.60006. [performance.md](performance.md) compares all backends against original Darknet using one protocol. Network timing includes transfer and synchronization; pipeline timing adds in-memory preprocessing and decode/NMS. Both exclude file I/O and model setup. The slight mAP differences are numerical/backend variation. The laptop native CUDA checkpoint scored 0.59801 validation / 0.58182 test with PPM inputs; Jetson evaluated JPEG inputs. These are this project's AP@0.5 calculations, not official VOC leaderboard scores.

## Historical external-Darknet checkpoint

The following commands and measurements use the earlier checkpoint trained by original Darknet. They remain as a baseline, not as evidence of native detector training.

### Rebuild from those weights

On a machine with Python, `numpy` and `onnx` installed (the optional exporter test also needs `onnxruntime`):

```sh
python3 native/tools/export_darknet_onnx.py \
  --cfg runs/yolov3-tiny-voc/yolov3-tiny-voc.cfg \
  --weights runs/yolov3-tiny-voc/final.weights \
  --output runs/yolov3-tiny-voc/final.onnx
python3 native/tests/test_export_darknet_onnx.py
```

The exporter refuses to overwrite an existing ONNX file. The observed export used the final 10,000-batch checkpoint with 320,000 images seen. Its input cfg SHA-256 was `fc1c950433a6db1354ec1d4bfa81ba6d86a4b6aac19d607cc249dfc7e525eb4b`, weights SHA-256 was `839fe2abd9a0e113104ff5acb3ff750f953d849ca3b0a8782536d330466a5c35`, and ONNX SHA-256 was `e77784d8b621763c299df7209c5cfcfb2ab2cb70d219f08cf49747dee6946d3e`. The ONNX checker and an ONNX Runtime operator smoke test passed. The exported full model was also parsed by TensorRT 10.3 and evaluated below.

Transfer `final.onnx` to the Jetson, then use its installed TensorRT `trtexec` to build engines **on that Jetson**. These are the observed TensorRT 10.3 commands; neither ONNX nor TensorRT is required for the original native CUDA path:

```sh
/usr/src/tensorrt/bin/trtexec \
  --onnx=/work/final.onnx --saveEngine=/work/final-fp32.engine \
  --memPoolSize=workspace:512 --skipInference

/usr/src/tensorrt/bin/trtexec \
  --onnx=/work/final.onnx --saveEngine=/work/final-fp16.engine \
  --fp16 --memPoolSize=workspace:512 --skipInference
```

Build the C++ engine runner with TensorRT enabled. `DETECTOR_TENSORRT=ON` is optional but requires `DETECTOR_CUDA=ON`; it links against the Jetson's installed TensorRT C++ headers/library. CPU and native CUDA builds can leave it off.

```sh
cmake -S . -B build-trt -DCMAKE_BUILD_TYPE=Release \
  -DDETECTOR_CUDA=ON -DDETECTOR_TENSORRT=ON -DDETECTOR_OPENCV=ON \
  -DCMAKE_CUDA_ARCHITECTURES=87 -DBUILD_TESTING=ON
cmake --build build-trt -j4
ctest --test-dir build-trt --output-on-failure

./build-trt/cuda_detector detect /work/yolov3-tiny-voc.cfg /work/final-fp16.engine \
  /work/image.jpg --backend tensorrt --names /work/voc.names
./build-trt/cuda_detector_eval /work/yolov3-tiny-voc.cfg /work/final-fp16.engine \
  /data/voc2007/images/test /data/voc2007/labels/test \
  --backend tensorrt --confidence 0.01
./build-trt/cuda_detector benchmark /work/yolov3-tiny-voc.cfg /work/final-fp16.engine \
  /work/image.ppm --backend tensorrt --scope pipeline --warmup 5 --iterations 30
```

For `--backend tensorrt`, the model argument is the `.engine`, **not** the Darknet `.weights`. The `.cfg` is still needed for the shared YOLO decoder. The runtime validates the engine's input/output names, shapes, and float32 I/O before inference. Its plan files are executable, hardware-/TensorRT-version-specific artifacts: only deserialize engines you built or trust, and rebuild on another device/version as needed. See NVIDIA's [TensorRT runtime guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/c-api-docs.html) and [TensorRT 10.x API migration](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/api/tensorrt-8x-to-10x-c-api.html).

### Historical Jetson results

All three backends used the same trained 20-class model, image, 416x416 input, existing Jetson Orin 25W power mode, native letterbox/decode/NMS, confidence 0.01 and NMS IoU 0.45 for AP evaluation. All validation and test images were processed (no `--limit`). The test split had 4,952 images and 12,032 non-difficult ground-truth objects.

| Backend | 501-image validation mAP@0.5 | 4,952-image test mAP@0.5 | Network mean | In-memory pipeline mean |
| --- | ---: | ---: | ---: | ---: |
| Native CUDA, Darknet `.weights` | 0.57379 | 0.55534 | 91.204 ms | 100.786 ms |
| TensorRT FP32 | 0.57344 | 0.55598 | 5.889 ms | 20.008 ms |
| TensorRT FP16 | 0.57535 | 0.55595 | 3.691 ms | 11.970 ms |

Timing used five warmups and 30 measured iterations on the same held-out PPM image with the GPU otherwise idle. Network time includes host-to-device transfer, execution, device-to-host transfer, and synchronization for each backend, but excludes preprocessing, decode/NMS, image-file I/O, weight/engine loading, and session setup. Pipeline time includes letterbox, network, decode/NMS, and inverse letterbox but still excludes image-file I/O and setup. The native CUDA convolution is untuned, so these are measured implementation results, not general YOLOv3-tiny or TensorRT speed claims. TensorRT FP16 and native CUDA each produced one `person` detection on the same held-out test image at scores 0.9966 and 0.9967, respectively.

The small AP differences are consistent with distinct floating-point execution. These are this project's all-point AP@0.5 scores on VOC labels converted **without** difficult objects; the evaluator does not implement VOC ignore regions, so they are not official VOC scores or COCO mAP@[.5:.95]. Per-class reports, build logs, engine files, and the ONNX model are in ignored `runs/yolov3-tiny-voc/`. The FP16/FP32 engines copied there are archives of **Jetson-built** plans, not portable Windows engines.
