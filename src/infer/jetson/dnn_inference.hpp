/**
 * @file trt_inference.hpppp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __TRT_INFERENCE_H_
#define __TRT_INFERENCE_H_
#include "NvInfer.h"
#include "inference.hpp"
#include "jetson/argsParser.h"
#include "jetson/buffers.h"
#include "jetson/common.h"
#include <NvInferRuntimeCommon.h>
#include <array>
#include <cmath>
#include <memory>
#include <opencv2/opencv.hpp>

using namespace infer::trt;

#define IMAGE_MAX_SIZE 1200 * 1200 * 3
namespace infer {
namespace dnn {
class AlgoInference : public Inference {
public:
  //!
  //! \brief construction
  //!
  AlgoInference(const AlgoBase &_param) : mParams(_param) {
    CHECK(cudaStreamCreate(&processStream));
    // prepare input data cache in pinned memory
    CHECK(cudaMallocHost((void **)&imageHost, IMAGE_MAX_SIZE));
    // prepare input data cache in device memory
    CHECK(cudaMalloc((void **)&imageDevice, IMAGE_MAX_SIZE));
  }

  //!
  //! \brief destruction
  //!
  ~AlgoInference() {
    cudaStreamDestroy(processStream);
    CHECK(cudaFree(imageDevice));
    CHECK(cudaFreeHost(imageHost));
    imageHost = nullptr;
    imageDevice = nullptr;
  }
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
  //! \brief Reads the input and mean data, preprocesses, and stores the result
  //! in a managed buffer
  //!
  virtual bool processInput(void *) override;

  //!
  //! \brief Outside can get model information after model initialize
  //!
  virtual void getModelInfo(ModelInfo &) const override;

private:
  uint8_t *imageHost = nullptr;
  uint8_t *imageDevice = nullptr;
  // Create stream
  cudaStream_t processStream;

private:
  //!< The TensorRT engine used to run the network
  std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};
  //!< The TensorRT execution context
  UniquePtr<nvinfer1::IExecutionContext> context{nullptr};

protected:
  //!< The parameters for the sample.
  AlgoBase mParams;
  //!< The dimensions of the input to the network.
  std::vector<Dims> inputDims;
  //!< The dimensions of the output to the network.
  std::vector<Dims> outputDims;

  std::shared_ptr<BufferManager> buffers;
};
} // namespace dnn
} // namespace infer
#endif
