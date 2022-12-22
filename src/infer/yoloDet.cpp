/**
 * @file yoloDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-07
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "yoloDet.hpp"
#include "logger/logger.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

namespace infer {
namespace vision {

void Yolo::generateBoxes(std::unordered_map<int, DetRet> &m,
                         void **outputs) const {
  float **output = reinterpret_cast<float **>(*outputs);
  for (int i = 0; i < modelInfo.output_count; ++i) {
    int numAnchors = modelInfo.outputShapes[i].at(1);
    int num = modelInfo.outputShapes[i].at(2);
    for (int j = 0; j < numAnchors * num; j += num) {
      if (output[i][j + 4] <= mParams.cond_thr)
        continue;
      // std::cout << output[i][j + 0] << ", " << output[i][j + 1] << ", "
      //           << output[i][j + 2] << ", " << output[i][j + 3] << ", "
      //           << output[i][j + 4] << ", " << output[i][j + 5] << std::endl;
      DetectionResult det;
      det.class_id = std::distance(
          output[i] + j + 5,
          std::max_element(output[i] + j + 5, output[i] + j + num));
      int real_idx = j + 5 + det.class_id;
      det.det_confidence = output[i][real_idx];
      memcpy(&det, &output[i][j], 5 * sizeof(float));
      if (m.count(det.class_id) == 0)
        m.emplace(det.class_id, DetRet());
      m[det.class_id].push_back(det);
    }
  }
}

FlowEngineModuleRegister(Yolo, const common::AlgorithmConfig &,
                         ModelInfo const &);

} // namespace vision
} // namespace infer