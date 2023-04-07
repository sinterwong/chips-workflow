/**
 * @file keypoints.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-04-04
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "keypoints.hpp"
#include "postprocess.hpp"
#include "vision.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

namespace infer::vision {

bool Keypoints::processInput(cv::Mat const &input, void **output,
                             common::ColorType) const {
  // 后面可以根据需求，实现基于opencv的预处理，比如resize和图片类型转换（bgr->rgb,
  // bgr->nv12, nv12->bgr..)
  return true;
}

bool Keypoints::processOutput(void **output, InferResult &result) const {
  std::unordered_map<int, KeypointsBoxes> cls2kbox;
  generateKeypointsBoxes(cls2kbox, output);
  return true;
}

bool Keypoints::verifyOutput(InferResult const &) const { return true; }
} // namespace infer::vision