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
#include "features.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

namespace infer {
namespace vision {

bool Features::processInput(cv::Mat const &input, void **output,
                            common::ColorType) const {
  // 后面可以根据需求，实现基于opencv的预处理，比如resize和图片类型转换（bgr->rgb,
  // bgr->nv12, nv12->bgr..)
  return true;
}

bool Features::processOutput(void **output, InferResult &result) const {
  return true;
}

bool Features::verifyOutput(InferResult const &) const { return true; }
} // namespace vision
} // namespace infer