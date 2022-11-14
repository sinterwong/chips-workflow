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
#include "infer_utils.hpp"
#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

namespace infer {
namespace vision {

bool Classifier::processInput(cv::Mat const &input, void **output, common::ColorType, common::ColorType) const {
  // 后面可以根据需求，实现基于opencv的预处理，比如resize和图片类型转换（bgr->rgb,
  // bgr->nv12, nv12->bgr..)
  return true;
}

bool Classifier::processOutput(void *output, Result &result) const {
  float *out = reinterpret_cast<float *>(output);
  
  int numClasses = modelInfo.outputShapes[0].at(1);
  result.classResult = softmax_argmax(out, numClasses);
  return true;
}

bool Classifier::verifyOutput(Result const &) const {
  return true;
}
} // namespace vision
} // namespace infer