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

#include "x3_inference.hpp"
#include "dnn/hb_dnn.h"
#include "logger/logger.hpp"
#include <algorithm>
#include <array>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace infer {
namespace x3 {

bool AlgoInference::initialize() {
  // 加载模型
  char const *model_path = mParams.modelPath.c_str();
  HB_CHECK_SUCCESS(hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path, 1),
                   "hbDNNInitializeFromFiles is failed!");

  // 获取模型名称
  char const **model_name_list;
  int model_count = 0;
  HB_CHECK_SUCCESS(
      hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
      "hbDNNGetModelNameList is failed!");

  // 获取dnn_handle
  HB_CHECK_SUCCESS(
      hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]),
      "hbDNNGetModelHandle is failed!");

  memset(&input_tensor, '\0', sizeof(hbDNNTensor));
  input_tensor.properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
  // 张量类型为Y通道及UV通道为输入的图片, 方便直接使用 vpu出来的y和uv分离的数据
  // 用于Y和UV分离的场景，主要为我们摄像头数据通路场景
  input_tensor.properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;

  // 准备模型输入数据（用于存放模型输入大小的数据）
  input_tensor_resized.properties.tensorLayout = HB_DNN_LAYOUT_NCHW;
  input_tensor_resized.properties.tensorType = HB_DNN_IMG_TYPE_NV12_SEPARATE;

  input_tensor_resized.sysMem[0].memSize =
      mParams.inputShape.at(1) * mParams.inputShape.at(0);
  hbSysMem &itr_mem0 = input_tensor_resized.sysMem[0];
  hbSysAllocMem(&itr_mem0, mParams.inputShape.at(1) * mParams.inputShape.at(0));
  input_tensor_resized.sysMem[1].memSize =
      mParams.inputShape.at(1) * mParams.inputShape.at(0) / 2;
  hbSysMem &itr_mem1 = input_tensor_resized.sysMem[1];
  hbSysAllocMem(&itr_mem1,
                mParams.inputShape.at(1) * mParams.inputShape.at(0) / 2);

  // NCHW
  input_tensor_resized.properties.validShape.numDimensions = 4;
  input_tensor_resized.properties.validShape.dimensionSize[0] = 1;
  input_tensor_resized.properties.validShape.dimensionSize[1] = 3;
  input_tensor_resized.properties.validShape.dimensionSize[2] =
      mParams.inputShape.at(1);
  input_tensor_resized.properties.validShape.dimensionSize[3] =
      mParams.inputShape.at(0);
  // 已满足对齐要求
  input_tensor_resized.properties.alignedShape =
      input_tensor_resized.properties.validShape;

  // 准备模型输出数据的空间
  int output_count;
  HB_CHECK_SUCCESS(hbDNNGetOutputCount(&output_count, dnn_handle),
                   "hbDNNGetOutputCount is failed!");
  output = new hbDNNTensor[output_count];

  for (int i = 0; i < output_count; i++) {
    std::vector<int> shape;
    hbDNNTensorProperties &output_properties = output[i].properties;
    HB_CHECK_SUCCESS(
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
        "hbDNNGetOutputTensorProperties is failed!");

    // 获取模型输出尺寸
    int out_aligned_size = 4;
    for (int j = 0; j < output_properties.alignedShape.numDimensions; j++) {
      out_aligned_size =
          out_aligned_size * output_properties.alignedShape.dimensionSize[j];
    }
    for (int j = 0; j < output_properties.validShape.numDimensions; j++) {
      shape.push_back(output_properties.validShape.dimensionSize[j]);
    }
    // for (auto i : shape) {
    //   std::cout << i << ", ";
    // }
    // std::cout << std::endl;
    outputShapes.push_back(shape);
    hbSysMem &mem = output[i].sysMem[0];
    HB_CHECK_SUCCESS(hbSysAllocCachedMem(&mem, out_aligned_size),
                     "hbSysAllocCachedMem is failed!");
  }

  return true;
}

bool AlgoInference::infer(void *inputs, void** outputs) {
  VIDEO_FRAME_S *stFrameInfo = reinterpret_cast<VIDEO_FRAME_S *>(inputs);
  // NV12 是 YUV420SP 格式
  input_tensor.sysMem[0].virAddr = stFrameInfo->stVFrame.vir_ptr[0];
  // input_tensor.sysMem[0].phyAddr = stFrameInfo->stVFrame.phy_ptr[0];
  input_tensor.sysMem[0].memSize =
      stFrameInfo->stVFrame.height * stFrameInfo->stVFrame.width;

  // 填充 input_tensor.data_ext 成员变量， UV 分量
  input_tensor.sysMem[1].virAddr = stFrameInfo->stVFrame.vir_ptr[1];
  // input_tensor.sysMem[1].phyAddr = stFrameInfo->stVFrame.phy_ptr[1];
  input_tensor.sysMem[1].memSize =
      stFrameInfo->stVFrame.height * stFrameInfo->stVFrame.width / 2;

  // HB_DNN_IMG_TYPE_NV12_SEPARATE 类型的 layout 为 (1, 3, h, w)
  input_tensor.properties.validShape.numDimensions = 4;
  input_tensor.properties.validShape.dimensionSize[0] = 1; // N
  input_tensor.properties.validShape.dimensionSize[1] = 3; // C
  input_tensor.properties.validShape.dimensionSize[2] =
      stFrameInfo->stVFrame.height; // H
  input_tensor.properties.validShape.dimensionSize[3] =
      stFrameInfo->stVFrame.width; // W
  input_tensor.properties.alignedShape =
      input_tensor.properties.validShape; // 已满足跨距对齐要求，直接赋值

  if (!processInput(nullptr)) {
    FLOWENGINE_LOGGER_ERROR("[AlgoInference::infer]: process input error!");
    return false;
  }

  // 推理模型
  hbDNNInferCtrlParam infer_ctrl_param;
  hbDNNTaskHandle_t infer_task_handle = nullptr;
  HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
  HB_CHECK_SUCCESS(hbDNNInfer(&infer_task_handle, &output,
                              &input_tensor_resized, dnn_handle,
                              &infer_ctrl_param),
                   "infer hbDNNInfer failed");

  // 等待任务结束
  HB_CHECK_SUCCESS(hbDNNWaitTaskDone(infer_task_handle, 0),
                   "infer hbDNNWaitTaskDone failed");
  // 释放推理task
  HB_CHECK_SUCCESS(hbDNNReleaseTask(infer_task_handle),
                   "infer hbDNNReleaseTask failed");
  // 解析模型输出
  hbSysFlushMem(&(output->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  *outputs = output->sysMem[0].virAddr;
  // outputs = reinterpret_cast<void *>(output->sysMem[0].virAddr);
  return true;
}

bool AlgoInference::processInput(void *) {

  hbDNNResizeCtrlParam ctrl;
  hbDNNTaskHandle_t resize_task_handle;
  HB_DNN_INITIALIZE_RESIZE_CTRL_PARAM(&ctrl);
  HB_CHECK_SUCCESS(hbDNNResize(&resize_task_handle, &input_tensor_resized,
                               &input_tensor, NULL, &ctrl),
                   "hbDNNResize failed");
  HB_CHECK_SUCCESS(hbDNNWaitTaskDone(resize_task_handle, 0),
                   "hbDNNWaitTaskDone failed");
  HB_CHECK_SUCCESS(hbDNNReleaseTask(resize_task_handle),
                   "hbDNNReleaseTask failed");

  return true;
}

void AlgoInference::getModelInfo(ModelInfo &info) const {
  info.output_count = output_count;
  info.outputShapes = outputShapes;
}

} // namespace x3
} // namespace infer
