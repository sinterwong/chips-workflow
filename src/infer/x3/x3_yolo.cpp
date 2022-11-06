
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
#include "x3_yolo.hpp"
#include <cassert>
#include <vector>

namespace infer {
namespace x3 {

void YoloDet::generateBoxes(
    std::unordered_map<int, std::vector<DetectionResult>> &m, void *buffer) const {
  // TODO output的信息还需要强化一下，比如说做一个结构体什么的
  float *output = reinterpret_cast<float *>(buffer);
  // std::cout << "outputShapes size: " << outputShapes.size() << std::endl;
  // std::cout << "outputShapes[0] size: " << outputShapes[0].size() << std::endl;
  assert(outputShapes[0].size() == 4);
  int numAnchors = outputShapes.at(0).at(1);
  int num = outputShapes.at(0).at(2);
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

bool YoloDet::verifyOutput(Result const &result) const { return true; }

} // namespace x3
} // namespace infer