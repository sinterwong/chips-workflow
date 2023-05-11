/**
 * @file dnn_inference.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "dnn_inference.hpp"
#include "logger/logger.hpp"
#include "preprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace infer {
namespace dnn {
using namespace infer::utils;

bool AlgoInference::initialize() { return true; }

bool AlgoInference::infer(FrameInfo &input, void **outputs) {
  // 预处理数据
  if (!processInput(reinterpret_cast<void *>(&input))) {
    FLOWENGINE_LOGGER_ERROR("[AlgoInference::infer]: process input error!");
    return false;
  }
  return true;
}

bool AlgoInference::processInput(void *inputs) { return true; }

void AlgoInference::getModelInfo(ModelInfo &info) const {}

} // namespace rockchip
} // namespace infer
