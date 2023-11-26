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
  engine = sp_init_bpu_module(mParams.modelPath.c_str());
  if (!engine) {
    FLOWENGINE_LOGGER_ERROR("sp_init_bpu_module is failed!");
  }

  // 准备模型输出结果
  HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, engine->m_dnn_handle),
                   "hbDNNGetOutputCount is failed!");
  output_tensor = new hbDNNTensor[output_count];
  sp_init_bpu_tensors(engine, output_tensor);

  for (int i = 0; i < output_count; i++) {
    std::vector<int> shape;
    // 获取模型输出尺寸
    for (int j = 0; j < output_tensor->properties.validShape.numDimensions;
         j++) {
      shape.push_back(output_tensor->properties.validShape.dimensionSize[j]);
    }
    outputShapes.push_back(shape);
  }

  engine->output_tensor = output_tensor;

  return true;
}

bool AlgoInference::infer(FrameInfo &input, void **outputs) {
  // 预处理数据
  if (!processInput(reinterpret_cast<void *>(&input))) {
    FLOWENGINE_LOGGER_ERROR("[AlgoInference::infer]: process input error!");
    return false;
  }

  // 推理模型
  sp_bpu_start_predict(engine, reinterpret_cast<char *>(input_data.data));
  // 解析模型输出
  hbSysFlushMem(&(output_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  float **ret = reinterpret_cast<float **>(*outputs);
  for (int i = 0; i < output_count; ++i) {
    ret[i] = reinterpret_cast<float *>(output_tensor[i].sysMem[0].virAddr);
  }
  return true;
}

bool AlgoInference::infer(cv::Mat const &input, void **outputs) {
  // 预处理数据
  input_data = input;
  // 推理模型
  sp_bpu_start_predict(engine, reinterpret_cast<char *>(input_data.data));
  // 解析模型输出
  hbSysFlushMem(&(output_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  float **ret = reinterpret_cast<float **>(*outputs);
  for (int i = 0; i < output_count; ++i) {
    ret[i] = reinterpret_cast<float *>(output_tensor[i].sysMem[0].virAddr);
  }
  return true;
}

bool AlgoInference::processInput(void *inputs) {
  auto inputData = reinterpret_cast<FrameInfo *>(inputs);
  char *data = reinterpret_cast<char *>(*inputData->data);
  int height = inputData->shape.at(1);
  int width = inputData->shape.at(0);
  input_data = cv::Mat(height * 3 / 2, width, CV_8UC1, data);

  // TODO to accomplish nv12 resize
  NV12toRGB(input_data, input_data);
  std::array<int, 2> shape = {mParams.inputShape.at(0),
                              mParams.inputShape.at(1)};
  if (!resizeInput(input_data, mParams.isScale, shape)) {
    FLOWENGINE_LOGGER_ERROR("utils::resizeInput is failed!");
    return false;
  }
  RGB2NV12(input_data, input_data);
  // cv::imwrite("temp.jpg", input_data);
  return true;
}

void AlgoInference::getModelInfo(ModelInfo &info) const {
  info.output_count = output_count;
  info.outputShapes = outputShapes;
}

} // namespace x3
} // namespace infer
