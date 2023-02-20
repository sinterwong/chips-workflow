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

namespace infer {
namespace vision {

class Classifier : public Vision {
  //!
  //! \brief construction
  //!
public:
  Classifier(const common::AlgorithmConfig &_param, ModelInfo const &_info)
      : Vision(_param, _info) {}

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processInput(cv::Mat const &input, void **output,
                            common::ColorType,
                            common::ColorType) const override;

  //!
  //! \brief Postprocessing that the output is correct and prints it
  //!
  virtual bool processOutput(void **, InferResult &) const override;

  //!
  //! \brief verifyOutput that the result is correct for infer
  //!
  virtual bool verifyOutput(InferResult const &) const override;

protected:
  virtual ClsRet generateClass(float *output) const = 0;
};
} // namespace vision
} // namespace infer

#endif