/**
 * @file classifier.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-14
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFERENCE_VISION_CLASSIFIER_H_
#define __INFERENCE_VISION_CLASSIFIER_H_
#include "logger/logger.hpp"
#include "vision.hpp"
#include <unordered_map>
#include <vector>

namespace infer {
namespace vision {

class Classifier : public Vision {
  //!
  //! \brief construction
  //!
public:
  Classifier(const common::AlgorithmConfig &_param, ModelInfo const &_info) : Vision(_param, _info) {}

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processInput(cv::Mat const &input, void **output,
                            common::ColorType,
                            common::ColorType) const override;

  //!
  //! \brief Postprocessing that the output is correct and prints it
  //!
  virtual bool processOutput(void **, Result &) const override;

  //!
  //! \brief verifyOutput that the result is correct for infer
  //!
  virtual bool verifyOutput(Result const &) const override;

private:
  ClsRet softmax_argmax(float *output, int outputSize) const {
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
};
} // namespace vision
} // namespace infer

#endif