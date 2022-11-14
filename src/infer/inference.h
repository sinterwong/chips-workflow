/**
 * @file inference.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __INFERENCE_H_
#define __INFERENCE_H_
#include "infer_common.hpp"
#include <array>

namespace infer {

class Inference {
public:
  virtual ~Inference() {}
  //!
  //! \brief initialize the network
  //!
  virtual bool initialize() = 0;

  //!
  //! \brief Runs the inference engine with input of void*
  //!
  virtual bool infer(FrameInfo &, void **) = 0;

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processInput(void *) = 0;

  //!
  //! \brief Outside can get model information after model initialize
  //!
  virtual void getModelInfo(ModelInfo &) const = 0;
};
} // namespace infer
#endif
