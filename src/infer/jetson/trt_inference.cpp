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

namespace infer {
namespace trt {
void hwc_to_chw(cv::InputArray src, cv::OutputArray dst) {
  const int src_h = src.rows();
  const int src_w = src.cols();
  const int src_c = src.channels();

  cv::Mat hw_c = src.getMat().reshape(1, src_h * src_w);

  const std::array<int, 3> dims = {src_c, src_h, src_w};
  dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));
  cv::Mat dst_1d = dst.getMat().reshape(1, {src_c, src_h, src_w});

  cv::transpose(hw_c, dst_1d);
}

void chw_to_hwc(cv::InputArray src, cv::OutputArray dst) {
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
  std::ifstream file(mParams.serializedFilePath, std::ios::binary);
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
  // inputDims = {mParams.batchSize, mParams.inputShape[2], mParams.inputShape[0],
  // mParams.inputShape[1]}; outputDims = {mParams.batchSize, mParams.numAnchors,
  // mParams.numClasses + 5}; context->setBindingDimensions(0, inputDims);
  // std::cout << context->getBindingDimensions(0) << std::endl;
  return true;
}

void TRTInference::infer(void* inputs, Result &result) {
  BufferManager buffers(mEngine, mParams.batchSize, context);

  if (!processInput(inputs, buffers)) {
    FLOWENGINE_LOGGER_ERROR("process input error!");
    exit(-1);
  }

  // buffers.copyInputToDevice();  // 以下的预处理直接将输入放进了device中
  // Memcpy from host input buffers to device input buffers
  bool status = context->executeV2(buffers.getDeviceBindings().data());
  if (!status) {
    FLOWENGINE_LOGGER_ERROR("execute error!");
    exit(-1);
  }

  // Memcpy from device output buffers to host output buffers
  buffers.copyOutputToHost();
  if (!processOutput(buffers, result)) {
    FLOWENGINE_LOGGER_ERROR("process output error!");
    exit(-1);
  }
}

bool TRTInference::processInput(void* inputs,
                                BufferManager const &buffers) const {
  float *deviceInputBuffer = static_cast<float *>(
      buffers.getDeviceBuffer(mParams.inputTensorNames[0])); // explicit batch
  int size_image =
      mParams.originShape[0] * mParams.originShape[1] * mParams.originShape[2];
  int size_image_dst =
      mParams.inputShape[0] * mParams.inputShape[1] * mParams.inputShape[2];
  // std::cout << mParams.originShape[0] << ", " << mParams.originShape[1] << std::endl;
  // std::cout << inputs[0].cols << ", " << inputs[0].rows << std::endl;
  for (int i = 0; i < mParams.batchSize; i++) {
    // copy data to pinned memory
    memcpy(imageHost, inputs, size_image);

    // copy data to device memory
    CHECK(cudaMemcpyAsync(imageDevice, imageHost, size_image,
                          cudaMemcpyHostToDevice, processStream));

    preprocess_kernel_img(imageDevice, mParams.originShape[0],
                          mParams.originShape[1], deviceInputBuffer,
                          mParams.inputShape[0], mParams.inputShape[1],
                          processStream);

    deviceInputBuffer += size_image_dst;
  }

  // cv::Mat resizedFrame;
  // if (mParams.scaling > 0) {
  //   int dw = mParams.originShape[0] * mParams.scaling;
  //   int dh = mParams.originShape[1] * mParams.scaling;
  //   cv::resize(inputs, resizedFrame, cv::Size(), mParams.scaling,
  //              mParams.scaling);
  //   cv::copyMakeBorder(
  //       resizedFrame, resizedFrame, 0, std::max(0, mParams.inputShape[1] -
  //       dh), 0, std::max(0, mParams.inputShape[0] - dw), cv::BORDER_WRAP,
  //       127);
  // } else {
  //   cv::resize(inputs, resizedFrame,
  //              cv::Size(mParams.inputShape[0], mParams.inputShape[1]));
  // }

  // cv::cvtColor(resizedFrame, resizedFrame, cv::COLOR_BGR2RGB);
  // // resizedFrame.convertTo(resizedFrame, CV_32FC3, 1.0 / 255);
  // hwc_to_chw(resizedFrame, resizedFrame);
  // cv::Mat inputImage = std::move(
  //     resizedFrame.reshape(1, resizedFrame.total() *
  //     resizedFrame.channels()));
  // std::vector<uint8_t> data = inputImage.isContinuous() ? inputImage :
  // inputImage.clone(); float *hostInputBuffer =
  //     static_cast<float
  //     *>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
  // // hostInputBuffer = data.data();  vector改成float就行，待会试试看能不能行
  // for (int i = 0; i < data.size(); i++) {
  //   hostInputBuffer[i] = float(data[i] / 255.0);
  // }
  return true;
}

bool TRTInference::verifyOutput(Result const &) const { return true; }

bool TRTInference::processOutput(BufferManager const &buffers,
                                 Result &result) const {
  return true;
}
} // namespace trt
} // namespace infer
