/**
 * @file detection.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-03
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "jetson/classifier.hpp"
#include "inference.h"
#include <NvInferPluginUtils.h>
#include <algorithm>
#include <math.h>
#include <utility>
#include <vector>

namespace infer {
namespace trt {

bool ClassifierInfer::verifyOutput(Result const &result) const { return true; }

std::pair<int, float> ClassifierInfer::softmax_argmax(float *output,
                                                      int outputSize) const {
  float val{0.0};
  int idx{0};

  // Calculate Softmax
  float sum{0.0};
  for (size_t i = 0; i < outputSize; i++) {
    output[i] = exp(output[i]);
    sum += output[i];
  }

  for (size_t i = 0; i < outputSize; i++) {
    output[i] /= sum; // 获取概率值
    if (val < output[i]) {
      val = output[i];
      idx = i;
    }
  }
  return std::pair<int, float>{idx, val};
}

bool ClassifierInfer::processOutput(BufferManager const &buffers,
                                    Result &result) const {
  float *hostOutputBuffer =
      static_cast<float *>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
  result.classResult =
      std::move(softmax_argmax(hostOutputBuffer, mParams.numClasses));
  return true;
};

} // namespace trt
} // namespace infer