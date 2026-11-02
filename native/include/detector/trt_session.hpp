#pragma once

#include "detector/network.hpp"
#include <memory>
#include <string>
#include <vector>

namespace detector {

// A Jetson-built TensorRT plan with the same raw YOLO/region head outputs (one
// per [yolo]/[region] layer in the network, in layer order) as the ONNX
// exporter; a cfg with no such layer that ends in a (de)convolutional layer
// (segmentation) maps to the single "logits" output. Preprocessing and decode/NMS remain in detector_core.
class TrtSession {
public:
    TrtSession(const Network& network, const std::string& engine_path);
    ~TrtSession();
    TrtSession(const TrtSession&) = delete;
    TrtSession& operator=(const TrtSession&) = delete;
    std::vector<Head> forward(const std::vector<float>& input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace detector
