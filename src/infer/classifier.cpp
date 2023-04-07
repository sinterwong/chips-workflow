/**
 * @file classifier.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-14
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "classifier.hpp"
#include "logger/logger.hpp"

namespace infer::vision {

bool Classifier::processInput(cv::Mat const &input, void **output,
                              common::ColorType) const {
  // 后面可以根据需求，实现基于opencv的预处理，比如resize和图片类型转换（bgr->rgb,
  // bgr->nv12, nv12->bgr..)
  return true;
}

bool Classifier::processOutput(void **output, InferResult &result) const {
  auto clsRet = generateClass(output);
  result.aRet = std::move(clsRet);
  return true;
}

bool Classifier::verifyOutput(InferResult const &) const { return true; }
} // namespace infer::vision