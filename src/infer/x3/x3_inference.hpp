/**
 * @file x3_inference.hpp
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

#include "dnn/hb_dnn.h"
#include "hb_comm_video.h"
#include "inference.h"
#include "logger/logger.hpp"
#include <array>
#include <cmath>
#include <memory>
#include <vector>
#include "common/config.hpp"

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

namespace infer {
namespace x3 {
class AlgoInference : public Inference {
public:
  //!
  //! \brief construction
  //!
  AlgoInference(const common::AlgorithmConfig &_param) : mParams(_param) {}

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

    // 释放HB内存
    // HB_CHECK_SUCCESS(hbSysFreeMem(&(input_tensor.sysMem[0])),
    //                  "hbSysFreeMem input_tensor.sysMem[0] failed!");
    HB_CHECK_SUCCESS(hbSysFreeMem(&(input_tensor_resized.sysMem[0])),
                     "hbSysFreeMem input_tensor_resized.sysMem[0] failed!");
    // for (int i = 0; i < output_count; i++) {
    //   HB_CHECK_SUCCESS(hbSysFreeMem(&(output[i].sysMem[0])),
    //                    "hbSysFreeMem output.sysMem[0] failed!");
    // }

    // 释放handle
    // HB_CHECK_SUCCESS(hbDNNReleaseTask(task_handle), "hbDNNReleaseTask
    // failed"); HB_CHECK_SUCCESS(hbDNNRelease(dnn_handle),
    //                  "hbDNNRelease dnn_handle failed");
    HB_CHECK_SUCCESS(hbDNNRelease(packed_dnn_handle),
                     "hbDNNRelease packed_dnn_handle failed");

    return 0;
  }

private:
  //!< The parameters for the sample.
  common::AlgorithmConfig mParams;
  // 准备输入数据（用于存放yuv数据）
  hbDNNTensor input_tensor;
  // resize后送给bpu运行的图像
  hbDNNTensor input_tensor_resized;

  // output
  hbDNNTensor *output;
  int output_count;
  std::vector<std::vector<int>> outputShapes;
  // dnn_handle
  hbDNNHandle_t dnn_handle;
  // 加载模型handle
  hbPackedDNNHandle_t packed_dnn_handle;
  // 用于储存输入数据
  uint64_t mmz_paddr[2];
  char *mmz_vaddr[2];
};
} // namespace x3
} // namespace infer
#endif
