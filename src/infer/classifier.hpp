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
  std::pair<int, float> softmax_argmax(float *output, int outputSize) const {
    float val{0.0};
    int idx{0};

    // Calculate Softmax
    float sum{0.0};
    for (int i = 0; i < outputSize; i++) {
      // std::cout << output[i] << std::endl;
      // output[i] = exp(output[i]);
      sum += output[i];
    }
    for (int i = 0; i < outputSize; i++) {
      output[i] /= sum; // 获取概率值
      if (val < output[i]) {
        val = output[i];
        idx = i;
      }
    }
    return std::pair<int, float>{idx, val};
  }
};
} // namespace vision
} // namespace infer

#endif