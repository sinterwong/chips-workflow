/**
 * @file dnn_inference.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __ONNXRUNTIME_INFERENCE_H_
#define __ONNXRUNTIME_INFERENCE_H_

#include "inference.hpp"
#include "logger/logger.hpp"

#include <array>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>

#include <onnxruntime_cxx_api.h>

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
    if (outputDatas != nullptr) {
      for (int i = 0; i < outputShapes.size(); i++) {
        if (outputDatas[i] != nullptr) {
          delete[] outputDatas[i];
          outputDatas[i] = nullptr;
        }
      }
      delete[] outputDatas;
      outputDatas = nullptr;
    }

    // 释放 outputNames
    if (outputNames != nullptr) {
      delete[] outputNames;
      outputNames = nullptr;
    }

    return 0;
  }

private:
  //!< The parameters for the sample.
  AlgoBase mParams;

  Ort::Value inputTensor{nullptr};
  std::vector<Ort::Value> outputTensors;

  // Ort::AllocatorWithDefaultOptions ortAlloc;

  const char *inputNames[1];
  std::vector<int64_t> inputShape;

  const char **outputNames;
  std::vector<std::vector<int64_t>> outputShapes;

  // infer engine
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> session;
  std::unique_ptr<Ort::MemoryInfo> memoryInfo;

  cv::Mat inputData;
  float **outputDatas;
};
} // namespace dnn
} // namespace infer
#endif
