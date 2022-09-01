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

#include "jetson/trt_inference.hpp"
#include "jetson/logger.h"
#include "jetson/preprocess.h"
#include "jetson/standard.h"
#include <algorithm>
#include <array>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace infer {
namespace trt {
template <typename T>
void chw_to_hwc(T *chw_data, T *hwc_data, int channel, int height, int width) {
  int wc = width * channel;
  int index = 0;
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        hwc_data[h * wc + w * channel + c] = chw_data[index];
        index++;
      }
    }
  }
}

template <typename T>
void hwc_to_chw(T *chw_data, T *hwc_data, int channel, int height, int width) {
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int c = 0; c < channel; c++) {
        chw_data[c * wh + h * width + w] = hwc_data[index];
        index++;
      }
    }
  }
}

bool TRTInference::initialize() {

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
    FLOWENGINE_LOGGER_WARN("CudaEngine is failed!");
    return false;
  }
  delete[] trtModelStream; // 创建完engine就没什么用了

  context =
      UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
  if (!context) {
    FLOWENGINE_LOGGER_WARN("ExecutionContext is failed!");
    return false;
  }

  // set the input and output dims
  for (auto &name : mParams.inputTensorNames) {
    auto index = mEngine->getBindingIndex(name.c_str());
    inputDims.push_back(mEngine->getBindingDimensions(index));
  }
  for (auto &name : mParams.outputTensorNames) {
    auto index = mEngine->getBindingIndex(name.c_str());
    outputDims.push_back(mEngine->getBindingDimensions(index));
  }
  return true;
}

bool TRTInference::infer(void *inputs, Result &result) {
  BufferManager buffers(mEngine, mParams.batchSize, context);

  if (!processInput(inputs, buffers, result.shape)) {
    FLOWENGINE_LOGGER_ERROR("process input error!");
    return false;
  }

  // buffers.copyInputToDevice();  // 以下的预处理直接将输入放进了device中
  // Memcpy from host input buffers to device input buffers
  bool status = context->executeV2(buffers.getDeviceBindings().data());
  if (!status) {
    FLOWENGINE_LOGGER_ERROR("execute error!");
    return false;
  }

  // Memcpy from device output buffers to host output buffers
  buffers.copyOutputToHost();
  if (!processOutput(buffers, result)) {
    FLOWENGINE_LOGGER_ERROR("process output error!");
    return false;
  }
  return true;
}

bool TRTInference::resizeInput(cv::Mat &image) const {
  if (mParams.isScale) {
    int height = image.rows;
    int width = image.cols;
    float ratio = std::min(mParams.inputShape[0] * 1.0 / width,
                           mParams.inputShape[1] * 1.0 / height);

    int dw = width * ratio;
    int dh = height * ratio;
    cv::resize(image, image, cv::Size(dw, dh));
    cv::copyMakeBorder(image, image, 0, std::max(0, mParams.inputShape[1] - dh),
                       0, std::max(0, mParams.inputShape[0] - dw),
                       cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
  } else {
    cv::resize(image, image,
               cv::Size(mParams.inputShape[0], mParams.inputShape[1]));
  }
  return true;
}

bool TRTInference::processInput(void *inputs, BufferManager const &buffers,
                                std::array<int, 3> const &shape) const {
  float *deviceInputBuffer = static_cast<float *>(
      buffers.getDeviceBuffer(mParams.inputTensorNames[0])); // explicit batch
  // /*
  cv::Mat image{shape[1], shape[0], CV_8UC3, inputs};

  int image_size = image.cols * image.rows * image.channels();

  if (!resizeInput(image)) {
    return false;
  }
  int resized_size = image.cols * image.rows * image.channels();

  hwc_to_chw(imageHost, image.data, image.channels(), image.rows, image.cols);
  // cv::Mat temp = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
  // 保存CHW图像（9宫格灰图）
  // hwc_to_chw(temp.data, image.data, image.channels(), image.rows, image.cols);
  // cv::imwrite("output.jpg", temp);

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

bool TRTInference::verifyOutput(Result const &) const { return true; }

bool TRTInference::processOutput(BufferManager const &buffers,
                                 Result &result) const {
  return true;
}
} // namespace trt
} // namespace infer
