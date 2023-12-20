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
  inputName = session->GetInputNameAllocated(0, ortAlloc).get();

  // 获取输出信息
  for (int i = 0; i < session->GetOutputCount(); i++) {
    auto outputInfo = session->GetOutputTypeInfo(i);
    outputShapes.push_back(outputInfo.GetTensorTypeAndShapeInfo().GetShape());
    outputNames.push_back(session->GetOutputNameAllocated(i, ortAlloc).get());
  }

  // 初始化输入数据
  inputData.create(inputShape.at(2), inputShape.at(3), CV_32FC3);

  inputTensor = Ort::Value::CreateTensor<float>(
      memoryInfo, reinterpret_cast<float *>(inputData.data),
      inputShape.at(0) * inputShape.at(1) * inputShape.at(2) * inputShape.at(3),
      inputShape.data(), inputShape.size());

  // 初始化输出数据
  for (int i = 0; i < outputShapes.size(); i++) {
    auto outputShape = outputShapes.at(i);
    size_t outputSize = 0;
    for (auto &dim : outputShape) {
      outputSize *= dim;
    }
    outputDatas.emplace_back(std::vector<float>(outputSize));
    outputTensors.emplace_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputDatas.at(i).data(), outputSize, outputShape.data(),
        outputShape.size()));
  }
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
  session->Run(runOptions, &inputName, &inputTensor, 1, outputNames.data(),
               outputTensors.data(), outputTensors.size());
  return true;
}

bool AlgoInference::infer(cv::Mat const &input, void **outputs) { return true; }

bool AlgoInference::processInput(void *inputs) { return true; }

void AlgoInference::getModelInfo(ModelInfo &info) const {}

} // namespace dnn
} // namespace infer
