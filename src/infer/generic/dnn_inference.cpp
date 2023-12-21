/**
 * @file dnn_inference.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "dnn_inference.hpp"
#include "logger/logger.hpp"
#include "preprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace infer {
namespace dnn {
using namespace infer::utils;

bool AlgoInference::initialize() {

  // 加载模型
  auto memoryInfo =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  session = std::make_unique<Ort::Session>(env, mParams.modelPath.c_str(),
                                           Ort::SessionOptions{nullptr});

  assert(session->GetInputCount() == 1);

  // 获取输入信息
  auto inputInfo = session->GetInputTypeInfo(0);
  assert(inputInfo.GetTensorTypeAndShapeInfo().GetDimensionsCount() == 4);
  inputShape = inputInfo.GetTensorTypeAndShapeInfo().GetShape();
  // inputName = session->GetInputNameAllocated(0, ortAlloc).get();
  inputNames[0] = mParams.inputNames.at(0).c_str();

  // 初始化输出名称
  // outputNames = new const char *[session->GetOutputCount()];

  // // 获取输出信息
  // for (int i = 0; i < session->GetOutputCount(); i++) {
  //   auto outputInfo = session->GetOutputTypeInfo(i);
  //   outputShapes.push_back(outputInfo.GetTensorTypeAndShapeInfo().GetShape());
  //   outputNames[i] = mParams.outputNames.at(i).c_str();
  // }
  outputNames[0] = mParams.outputNames.at(0).c_str();

  // 不用初始化输入数据，因为后面会直接指向外部传入的输入数据
  // inputData.create(1,
  //                  inputShape.at(0) * inputShape.at(1) * inputShape.at(2) *
  //                      inputShape.at(3),
  //                  CV_32FC1);

  inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, reinterpret_cast<float *>(inputData.data),
      inputShape.at(0) * inputShape.at(1) * inputShape.at(2) * inputShape.at(3),
      inputShape.data(), inputShape.size());

  // // 初始化输出数据
  // outputDatas = new float *[outputShapes.size()];
  // for (int i = 0; i < outputShapes.size(); i++) {
  //   auto outputShape = outputShapes.at(i);
  //   size_t outputSize = 1;
  //   for (auto &dim : outputShape) {
  //     outputSize *= dim;
  //   }
  //   outputDatas[i] = new float[outputSize];
  //   outputTensors.emplace_back(Ort::Value::CreateTensor<float>(
  //       memoryInfo, outputDatas[i], outputSize, outputShape.data(),
  //       outputShape.size()));
  // }
  // auto outputShape = outputShapes.at(0);

  auto outputInfo = session->GetOutputTypeInfo(0);
  auto outputShape = outputInfo.GetTensorTypeAndShapeInfo().GetShape();
  size_t outputSize = 1;
  for (auto &dim : outputShape) {
    outputSize *= dim;
  }
  // outputDatas[0] = new float[outputSize];
  // outputTensor = Ort::Value::CreateTensor<float>(
  //     memoryInfo, outputDatas[0], outputSize, outputShapes.at(0).data(),
  //     outputShapes.at(0).size());

  outputDatas = new float[outputSize];
  outputTensor =
      Ort::Value::CreateTensor<float>(memoryInfo, outputDatas, outputSize,
                                      outputShape.data(), outputShape.size());
  return true;
}

bool AlgoInference::infer(FrameInfo &input, void **outputs) {
  // 预处理数据
  if (!processInput(reinterpret_cast<void *>(&input))) {
    FLOWENGINE_LOGGER_ERROR("process input error!");
    return false;
  }

  // 执行推理
  Ort::RunOptions runOptions;
  // session->Run(runOptions, inputName, &inputTensor, 1, outputNames,
  //              outputTensors.data(), outputTensors.size());

  session->Run(runOptions, inputNames, &inputTensor, 1, outputNames,
               &outputTensor, 1);

  // //
  // TODO:这里指向的是类内的outputDatas，因为数据没有拷贝出去，所以后处理当前不能在外面并发完成
  // float **ret = reinterpret_cast<float **>(outputs);
  // for (int i = 0; i < outputShapes.size(); i++) {
  //   ret[i] = outputDatas[i];
  // }
  float **ret = reinterpret_cast<float **>(outputs);
  ret[0] = outputTensor.GetTensorMutableData<float>();
  return true;
}

bool AlgoInference::infer(cv::Mat const &input, void **outputs) {

  // 数据已经完成了resize和chw转换
  inputData = input;

  // normalize
  inputData.convertTo(inputData, CV_32FC3, mParams.alpha, mParams.beta);

  // 执行推理
  Ort::RunOptions runOptions;
  // session->Run(runOptions, inputName, &inputTensor, 1, outputNames,
  //              outputTensors.data(), outputTensors.size());
  session->Run(runOptions, inputNames, &inputTensor, 1, outputNames,
               &outputTensor, 1);

  // TODO:这里指向的是类内的outputDatas，因为数据没有拷贝出去，所以后处理当前不能在外面并发完成
  float **ret = reinterpret_cast<float **>(outputs);
  // for (int i = 0; i < outputShapes.size(); i++) {
  //   ret[i] = outputDatas[i];
  // }
  ret[0] = outputTensor.GetTensorMutableData<float>();
  return true;
}

bool AlgoInference::processInput(void *inputs) {
  auto ImageData = reinterpret_cast<FrameInfo *>(inputs);
  cv::Mat image{ImageData->shape[1], ImageData->shape[0], CV_8UC3,
                ImageData->data};
  std::array<int, 2> shape = {mParams.inputShape.at(0),
                              mParams.inputShape.at(1)};

  if (!utils::resizeInput(image, mParams.isScale, shape)) {
    FLOWENGINE_LOGGER_ERROR("resize input error!");
    return false;
  }

  // normalize
  image.convertTo(image, CV_32FC3, mParams.alpha, mParams.beta);

  // hwc to chw
  utils::hwc_to_chw(reinterpret_cast<float *>(inputData.data),
                    reinterpret_cast<float *>(image.data), image.channels(),
                    image.rows, image.cols);
  return true;
}

void AlgoInference::getModelInfo(ModelInfo &info) const {
  info.output_count = outputShapes.size();
  for (auto &shape : outputShapes) {
    std::vector<int> tmp;
    for (auto &dim : shape) {
      tmp.push_back(dim);
    }
    info.outputShapes.emplace_back(tmp);
  }
}

} // namespace dnn
} // namespace infer
