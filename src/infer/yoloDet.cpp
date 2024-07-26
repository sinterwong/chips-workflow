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

namespace infer::vision {

void Yolo::generateBoxes(std::unordered_map<int, BBoxes> &m,
                         void **outputs) const {
  float **output = reinterpret_cast<float **>(outputs);
  float *out = output[0]; // just one output
  int numAnchors = modelInfo.outputShapes[0].at(1);
  int num = modelInfo.outputShapes[0].at(2);
  for (int j = 0; j < numAnchors * num; j += num) {
    if (out[j + 4] <= config.cond_thr)
      continue;
    BBox det;
    det.class_id = std::distance(out + j + 5,
                                 std::max_element(out + j + 5, out + j + num));
    int real_idx = j + 5 + det.class_id;
    det.det_confidence = out[real_idx];
    memcpy(&det, &out[j], 5 * sizeof(float));
    if (m.count(det.class_id) == 0)
      m.emplace(det.class_id, BBoxes());
    m[det.class_id].push_back(det);
  }
}
} // namespace infer::vision