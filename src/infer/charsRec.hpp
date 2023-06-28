/**
 * @file charsRec.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __INFERENCE_VISION_OCR_H_
#define __INFERENCE_VISION_OCR_H_
#include "vision.hpp"

namespace infer::vision {

class CharsRec : public Vision {
  //!
  //! \brief construction
  //!
public:
  explicit CharsRec(const AlgoConfig &_param, ModelInfo const &_info)
      : Vision(_param, _info) {
    config = mParams.getCopyParams<ClassAlgo>(); // 未来可能需要新的配置类型
  }

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processInput(cv::Mat const &input, void **output,
                            common::ColorType) const override {
    return true;
  }

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
  virtual CharsRet generateChars(void **output) const = 0;
};
} // namespace infer::vision

#endif