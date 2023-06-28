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

bool Classifier::processOutput(void **output, InferResult &result) const {
  auto clsRet = generateClass(output);
  result.aRet = std::move(clsRet);
  return true;
}

bool Classifier::verifyOutput(InferResult const &) const { return true; }
} // namespace infer::vision