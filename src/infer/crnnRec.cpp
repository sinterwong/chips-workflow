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
#include "crnnRec.hpp"
#include "charsRec.hpp"
#include "logger/logger.hpp"
#include "utils/factory.hpp"

#include <algorithm>

namespace infer::vision {

CharsRet CRNN::decodeChars(CharsRet const &preds) const {
  CharsRet ret;
  int pre = 0;
  for (size_t i = 0; i < preds.size(); ++i) {
    if (preds[i] != 0 && preds[i] != pre) {
      ret.push_back(preds[i]);
    }
    pre = preds[i];
  }
  return ret;
}

CharsRet CRNN::generateChars(void **outputs) const {
  float **out = reinterpret_cast<float **>(outputs);
  float *output = out[0]; // just one output
  int numChars = modelInfo.outputShapes[0].at(1);
  int numClasses = modelInfo.outputShapes[0].at(2);
  std::vector<int> predIds;
  for (int i = 0; i < numChars; ++i) {
    int idx = std::distance(output + i * numClasses,
                            std::max_element(output + i * numClasses,
                                             output + (i + 1) * numClasses));
    predIds.push_back(idx);
  }
  return decodeChars(predIds);
}
} // namespace infer::vision