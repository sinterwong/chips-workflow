/**
 * @file features.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "features.hpp"

namespace infer::vision {

bool Features::processOutput(void **output, InferResult &result) const {

  Eigenvector feature;
  generateFeature(output, feature);
  result.aRet = std::move(feature);
  return true;
}

bool Features::verifyOutput(InferResult const &) const { return true; }
} // namespace infer::vision