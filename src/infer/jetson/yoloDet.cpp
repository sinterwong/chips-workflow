
/**
 * @file yoloDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-28
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "jetson/yoloDet.hpp"
#include <cassert>

namespace infer {
namespace trt {

void YoloDet::generateBoxes(
    std::unordered_map<int, std::vector<DetectionResult>> &m,
    BufferManager const &buffers) const {


  for (int i = 0; i < mParams.outputTensorNames.size(); ++i) {
    float *output = static_cast<float *>(
        buffers.getHostBuffer(mParams.outputTensorNames[i]));

    assert(outputDims[i].nbDims == 3);
    int numAnchors = outputDims[i].d[1];
    int num = numAnchors + 5;
    for (int i = 0; i < numAnchors * num; i += num) {
      if (output[i + 4] <= mParams.cond_thr)
        continue;
      DetectionResult det;
      det.class_id = std::distance(
          output + i + 5, std::max_element(output + i + 5, output + i + num));
      int real_idx = i + 5 + det.class_id;
      det.det_confidence = output[real_idx];
      memcpy(&det, &output[i], 5 * sizeof(float));
      if (m.count(det.class_id) == 0)
        m.emplace(det.class_id, std::vector<DetectionResult>());
      m[det.class_id].push_back(det);
    }
  }
}

bool YoloDet::verifyOutput(Result const &result) const { return true; }

} // namespace trt
} // namespace infer