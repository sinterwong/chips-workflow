/**
 * @file infer_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFERENCE_UTILS_H_
#define __INFERENCE_UTILS_H_
#include "infer_common.hpp"
#include <array>
#include <unordered_map>
#include <vector>

namespace infer {
namespace utils {
inline bool compare(DetectionResult const &a, DetectionResult const &b) {
  return a.det_confidence > b.det_confidence;
}

float iou(std::array<float, 4> const &, std::array<float, 4> const &);

void nms(std::vector<DetectionResult> &,
         std::unordered_map<int, std::vector<DetectionResult>> &, float);

void renderOriginShape(std::vector<DetectionResult> &results,
                       std::array<int, 3> const &shape,
                       std::array<int, 3> const &inputShape, bool isScale);

} // namespace utils
} // namespace infer
#endif
