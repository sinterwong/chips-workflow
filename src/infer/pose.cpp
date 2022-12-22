/**
 * @file pose.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include "pose.hpp"
#include "infer_utils.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

namespace infer {
namespace vision {

bool Pose::processInput(cv::Mat const &input, void **output,
                             common::ColorType, common::ColorType) const {
  // 后面可以根据需求，实现基于opencv的预处理，比如resize和图片类型转换（bgr->rgb,
  // bgr->nv12, nv12->bgr..)
  return true;
}

bool Pose::processOutput(void **output, Result &result) const {
  return true;
}

bool Pose::verifyOutput(Result const &) const { return true; }
} // namespace vision
} // namespace infer