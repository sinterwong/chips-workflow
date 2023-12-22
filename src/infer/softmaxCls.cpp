/**
 * @file softmaxCls.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "softmaxCls.hpp"
#include "common/factory.hpp"
#include "logger/logger.hpp"

namespace infer::vision {

ClsRet Softmax::generateClass(void **outputs) const {
  float **out = reinterpret_cast<float **>(outputs);
  float *output = out[0]; // just one output

  int outputSize = modelInfo.outputShapes[0].at(1);

  float val{0.0};
  int idx{0};

  // Calculate Softmax
  float sum = 0.;
  // FLOWENGINE_LOGGER_INFO("outputSize: {}", outputSize);
  for (int i = 0; i < outputSize; i++) {
    // FLOWENGINE_LOGGER_INFO("before val: {}", output[i]);
    output[i] = std::exp(output[i]);
    // FLOWENGINE_LOGGER_INFO("after val: {}", output[i]);
    sum += output[i];
  }
  // FLOWENGINE_LOGGER_INFO("**********");
  for (int i = 0; i < outputSize; i++) {
    output[i] /= sum; // 获取概率值
    if (val < output[i]) {
      val = output[i];
      idx = i;
    }
  }
  return ClsRet{idx, val};
}

FlowEngineModuleRegister(Softmax, AlgoConfig const &, ModelInfo const &);
} // namespace infer::vision