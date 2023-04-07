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
#include <memory>

namespace infer::vision {

class Classifier : public Vision {
  //!
  //! \brief construction
  //!
public:
  explicit Classifier(const AlgoConfig &_param, ModelInfo const &_info)
      : Vision(_param, _info) {
    config = mParams.getCopyParams<ClassAlgo>();
  }

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processInput(cv::Mat const &input, void **output,
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
  ClassAlgo config;
  virtual ClsRet generateClass(void **output) const = 0;
};
} // namespace infer::vision

#endif