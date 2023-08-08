/**
 * @file trt_inference.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "dnn_inference.hpp"
#include "jetson/logger.h"
#include "jetson/preprocess.h"
#include "jetson/standard.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iterator>
#include <memory>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "infer/preprocess.hpp"
#include "logger/logger.hpp"

namespace infer {
namespace dnn {

bool AlgoInference::initialize() {

  std::shared_ptr<IRuntime> runtime(
      createInferRuntime(infer::trt::sample::gLogger));
  if (!runtime) {
    FLOWENGINE_LOGGER_WARN("Runtime is failed!");
    return false;
  }
  char *trtModelStream{nullptr};
  size_t size{0};
  std::ifstream file(mParams.modelPath, std::ios::binary);
  if (file.good()) {
    // LOG(INFO) << "Loading the engine: " << mParams.serializedFilePath
    // << std::endl;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
  }

  mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(trtModelStream, size));
  if (!mEngine) {
    FLOWENGINE_LOGGER_ERROR("CudaEngine is failed!");
    return false;
  }
  delete[] trtModelStream; // 创建完engine就没什么用了

  context =
      UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
  if (!context) {
    FLOWENGINE_LOGGER_ERROR("ExecutionContext is failed!");
    return false;
  }

  // set the input and output dims
  for (auto &name : mParams.inputNames) {
    auto index = mEngine->getBindingIndex(name.c_str());
    if (index < 0) {
      FLOWENGINE_LOGGER_ERROR("Cannot find binding of given name {} in {}",
                              name, mParams.modelPath);
      return false;
    }
    inputDims.push_back(mEngine->getBindingDimensions(index));
  }
  for (auto &name : mParams.outputNames) {
    auto index = mEngine->getBindingIndex(name.c_str());
    if (index < 0) {
      FLOWENGINE_LOGGER_ERROR("Cannot find binding of given name {} in {}",
                              name, mParams.modelPath);
      return false;
    }
    outputDims.push_back(mEngine->getBindingDimensions(index));
  }
  buffers =
      std::make_shared<BufferManager>(mEngine, mParams.batchSize, context);
  return true;
}

bool AlgoInference::infer(FrameInfo &inputs, void **outputs) {
  if (!processInput(reinterpret_cast<void *>(&inputs))) {
    FLOWENGINE_LOGGER_ERROR("process input error!");
    return false;
  }

  // buffers.copyInputToDevice();  // 以下的预处理直接将输入放进了device中
  // Memcpy from host input buffers to device input buffers
  bool status = context->executeV2(buffers->getDeviceBindings().data());
  if (!status) {
    FLOWENGINE_LOGGER_ERROR("execute error!");
    return false;
  }

  // Memcpy from device output buffers to host output buffers
  buffers->copyOutputToHost();

  float **output = reinterpret_cast<float **>(*outputs);
  for (int i = 0; i < mParams.outputNames.size(); ++i) {
    output[i] =
        static_cast<float *>(buffers->getHostBuffer(mParams.outputNames[i]));
  }
  return true;
}

bool AlgoInference::processInput(void *inputs) {
  float *deviceInputBuffer = static_cast<float *>(
      buffers->getDeviceBuffer(mParams.inputNames[0])); // explicit batch
  auto inputData = reinterpret_cast<FrameInfo *>(inputs);
  cv::Mat image{inputData->shape[1], inputData->shape[0], CV_8UC3,
                *inputData->data};
  // cv::imwrite("temp_out.jpg", image);

  std::array<int, 2> shape = {mParams.inputShape.at(0),
                              mParams.inputShape.at(1)};
  if (!utils::resizeInput(image, mParams.isScale, shape)) {
    return false;
  }
  int resized_size = image.cols * image.rows * image.channels();

  utils::hwc_to_chw(imageHost, image.data, image.channels(), image.rows,
                    image.cols);
  // cv::Mat temp = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  // 保存CHW图像（9宫格灰图）
  // hwc_to_chw(temp.data, image.data, image.channels(), image.rows,
  // image.cols); cv::imwrite("output.jpg", temp);

  // copy data to device memory
  CHECK(cudaMemcpyAsync(imageDevice, imageHost, resized_size,
                        cudaMemcpyHostToDevice, processStream));
  strandard_image(imageDevice, deviceInputBuffer, resized_size, mParams.alpha,
                  mParams.beta, processStream);

  // if batch_size > 1
  // int size_image_dst = mParams.inputShape[0] * mParams.inputShape[1] *
  // mParams.inputShape[2]; deviceInputBuffer += size_image_dst;
  // */

  /*
  memcpy(imageHost, inputs, size_image);

  // copy data to device memory
  CHECK(cudaMemcpyAsync(imageDevice, imageHost, size_image,
                        cudaMemcpyHostToDevice, processStream));

  preprocess_kernel_img(imageDevice, shape[0], shape[1], deviceInputBuffer,
                        mParams.inputShape[0], mParams.inputShape[1],
                        mParams.alpha, mParams.beta, processStream);
  */

  return true;
}

void AlgoInference::getModelInfo(ModelInfo &info) const {
  std::vector<std::vector<int>> outputShapes;
  for (auto &dim : outputDims) {
    std::vector<int> shape{dim.d, dim.d + dim.nbDims};
    outputShapes.emplace_back(shape);
  }
  info.output_count = mParams.outputNames.size();
  info.outputShapes = std::move(outputShapes);
}
} // namespace dnn
} // namespace infer
