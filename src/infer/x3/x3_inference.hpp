/**
 * @file trt_inference.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __X3_INFERENCE_H_
#define __X3_INFERENCE_H_

#define IMAGE_MAX_SIZE 1200 * 1200 * 3
#include "inference.h"
#include <array>
#include <cmath>
#include <memory>

namespace infer {
namespace x3 {
class X3Inference : public Inference {
public:
  //!
  //! \brief construction
  //!
  X3Inference(const common::AlgorithmConfig &_param) : mParams(_param) {}

  //!
  //! \brief destruction
  //!
  ~X3Inference() {}
  //!
  //! \brief initialize the network
  //!
  virtual bool initialize() override;

  //!
  //! \brief Runs the inference engine
  //!
  virtual bool infer(void *, Result &) override;
  // virtual bool infer(cv::Mat const&, Result &) override;

protected:
  //!< The parameters for the sample.
  common::AlgorithmConfig mParams;
};
} // namespace x3
} // namespace infer
#endif
