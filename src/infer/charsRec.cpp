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

namespace infer::vision {

bool CharsRec::processOutput(void **output, InferResult &result) const {
  auto ret = generateChars(output);
  result.aRet = std::move(ret);
  return true;
}

bool CharsRec::verifyOutput(InferResult const &) const { return true; }
} // namespace infer::vision