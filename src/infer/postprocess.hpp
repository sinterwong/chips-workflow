/**
 * @file postprocess.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __INFERENCE_UTILS_POSTPROCESS_H_
#define __INFERENCE_UTILS_POSTPROCESS_H_
#include "infer_common.hpp"
#include "opencv2/imgproc.hpp"

namespace infer {
namespace utils {

inline bool compare(BBox const &a, BBox const &b) {
  return a.det_confidence > b.det_confidence;
}

float iou(std::array<float, 4> const &, std::array<float, 4> const &);

void nms(BBoxes &, std::unordered_map<int, BBoxes> &, float);

// 复原bbox原图尺寸
void restoryBoxes(BBoxes &results, Shape const &shape,
                  Shape const &inputShape, bool isScale);

// 复原points原始尺寸
void restoryPoints(Points &results, Shape const &shape,
                   Shape const &inputShape, bool isScale);
} // namespace utils
} // namespace infer
#endif