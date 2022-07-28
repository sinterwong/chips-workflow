/**
 * @file detection.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-03
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __INFERENCE_CLASSIFIER_H_
#define __INFERENCE_CLASSIFIER_H_
#include "inference.h"
#include "jetson/trt_inference.hpp"

namespace infer {
namespace trt {
class ClassifierInfer : public TRTInference {
  //!
  //! \brief construction
  //!
public:
  ClassifierInfer(const common::AlgorithmConfig &_param) : TRTInference(_param) {}

private:
  //!
  //! \brief Verifies that the output is correct and prints it
  //!
  virtual bool verifyOutput(Result const &) const override;

  //!
  //! \brief Postprocessing that the output is correct and prints it
  //!
  virtual bool processOutput(BufferManager const &,
                             Result &) const override;

  //!
  //! \brief Reads the input and mean data, preprocesses, and stores the result
  //! in a managed buffer
  //!
  // virtual bool processInput(void *, BufferManager const &) const override;

  //!
  //! \brief Softmax and argmax for output
  //!
  std::pair<int, float> softmax_argmax(float* output, int outputSize) const;

};
} // namespace trt
} // namespace infer

#endif