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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace infer {
namespace trt {
void hwc_to_chw(cv::InputArray &src, cv::OutputArray &dst) {
  const int src_h = src.rows();
  const int src_w = src.cols();
  const int src_c = src.channels();

  cv::Mat hw_c = src.getMat().reshape(1, src_h * src_w);

  const std::array<int, 3> dims = {src_c, src_h, src_w};
  dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));
  cv::Mat dst_1d = dst.getMat().reshape(1, {src_c, src_h, src_w});

  cv::transpose(hw_c, dst_1d);
}

void chw_to_hwc(cv::InputArray &src, cv::OutputArray &dst) {
  const auto &src_size = src.getMat().size;
  const int src_c = src_size[0];
  const int src_h = src_size[1];
  const int src_w = src_size[2];

  auto c_hw = src.getMat().reshape(0, {src_c, src_h * src_w});

  dst.create(src_h, src_w, CV_MAKETYPE(src.depth(), src_c));
  cv::Mat dst_1d = dst.getMat().reshape(src_c, {src_h, src_w});

  cv::transpose(c_hw, dst_1d);
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
  std::cout << mEngine->getBindingDimensions(0) << std::endl;
  delete[] trtModelStream; // 创建完engine就没什么用了

  context =
      UniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
  if (!context) {
    FLOWENGINE_LOGGER_WARN("ExecutionContext is failed!");
    return false;
  }
  // inputDims = {mParams.batchSize, mParams.inputShape[2],
  // mParams.inputShape[0], mParams.inputShape[1]}; outputDims =
  // {mParams.batchSize, mParams.numAnchors, mParams.numClasses + 5};
  // context->setBindingDimensions(0, inputDims); std::cout <<
  // context->getBindingDimensions(0) << std::endl;
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
                       cv::BORDER_CONSTANT, 0);
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
  // cv::imwrite("/home/wangxt/workspace/projects/flowengine/tests/data/out.jpg", image);
  int image_size = image.cols * image.rows * image.channels();

  if (!resizeInput(image)) {
    return false;
  }
  int resized_size = image.cols * image.rows * image.channels();

  cv::Mat chw_image;

  hwc_to_chw(image, chw_image);
  
  memcpy(imageHost, chw_image.data, resized_size + 1);

  // copy data to device memory
  CHECK(cudaMemcpyAsync(imageDevice, imageHost, resized_size,
                        cudaMemcpyHostToDevice, processStream));
  strandard_image(imageDevice, deviceInputBuffer, resized_size, mParams.alpha, mParams.beta, processStream);

  // if batch_size > 1
  // int size_image_dst = mParams.inputShape[0] * mParams.inputShape[1] * mParams.inputShape[2]; 
  // deviceInputBuffer += size_image_dst;
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
