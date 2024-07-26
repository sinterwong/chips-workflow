/**
 * @file dnn_inference.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef __RKNN_INFERENCE_H_
#define __RKNN_INFERENCE_H_

#include "inference.hpp"
#include "logger/logger.hpp"


#include <opencv2/core/mat.hpp>
#include <memory>

namespace infer {
namespace dnn {
class AlgoInference : public Inference {
public:
  //!
  //! \brief construction
  //!
  AlgoInference(const AlgoBase &_param) : mParams(_param) {}

  //!
  //! \brief destruction
  //!
  ~AlgoInference() { terminate(); }
  //!
  //! \brief initialize the network
  //!
  virtual bool initialize() override;

  //!
  //! \brief Runs the inference engine
  //!
  virtual bool infer(FrameInfo &, void **) override;

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  bool processInput(void *) override;

  //!
  //! \brief Outside can get model information after model initialize
  //!
  virtual void getModelInfo(ModelInfo &) const override;

private:
  //!
  //! \brief Release the resource
  //!
  int terminate() {
    // len代表单次推理的输出数量
    return 0;
  }

private:
  //!< The parameters for the sample.
  AlgoBase mParams;
};
} // namespace x3
} // namespace infer
#endif
