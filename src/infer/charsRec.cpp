/**
 * @file charsRec.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-03-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "charsRec.hpp"
#include "logger/logger.hpp"

namespace infer {
namespace vision {

bool CharsRec::processInput(cv::Mat const &input, void **output,
                              common::ColorType) const {
  // 后面可以根据需求，实现基于opencv的预处理，比如resize和图片类型转换（bgr->rgb,
  // bgr->nv12, nv12->bgr..)
  return true;
}

bool CharsRec::processOutput(void **output, InferResult &result) const {
  auto ret = generateChars(output);
  result.aRet = std::move(ret);
  return true;
}

bool CharsRec::verifyOutput(InferResult const &) const { return true; }
} // namespace vision
} // namespace infer