/**
 * @file dnn_inference.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __X3_INFERENCE_H_
#define __X3_INFERENCE_H_

#include "inference.h"
#include "logger/logger.hpp"

#include <memory>
#include <opencv2/core/mat.hpp>
#include <sp_bpu.h>

#define HB_CHECK_SUCCESS(value, errmsg)                                        \
  do {                                                                         \
    /*value can be call of function*/                                          \
    int ret_code = value;                                                      \
    if (ret_code != 0) {                                                       \
      FLOWENGINE_LOGGER_ERROR("[BPU ERROR] {}, error code:{}", errmsg,         \
                              ret_code);                                       \
      return false;                                                            \
    }                                                                          \
  } while (0);

#define MAX_SIZE 2560 * 1440
#define BUFFER_NUM 2

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

  virtual bool infer(cv::Mat const &, void **) override;

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
    sp_deinit_bpu_tensor(output_tensor, output_count);
    sp_release_bpu_module(engine);
    if (output_tensor) {
      delete[] output_tensor;
    }
    return 0;
  }

private:
  //!< The parameters for the sample.
  AlgoBase mParams;
  // output
  int output_count;
  hbDNNTensor *output_tensor;
  std::vector<std::vector<int>> outputShapes;

  // infer engine
  bpu_module *engine;

  // input data
  cv::Mat input_data;
};
} // namespace dnn
} // namespace infer
#endif
