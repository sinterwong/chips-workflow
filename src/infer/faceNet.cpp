/**
 * @file faceNet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-04-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "faceNet.hpp"

namespace infer::vision {
void FaceNet::generateFeature(void **outputs, Eigenvector &feature) const {
  float **out = reinterpret_cast<float **>(*outputs);
  float *output = out[0];

  // 特征
  for (int i = 0; i < config->dim; ++i) {
    feature.push_back(output[i]);
  }
}

FlowEngineModuleRegister(FaceNet, AlgoConfig const &, ModelInfo const &);
}