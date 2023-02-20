/**
 * @file infer_common.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "common/common.hpp"
#include <array>
#include <vector>

#ifndef __INFERENCE_COMMON_H_
#define __INFERENCE_COMMON_H_

namespace infer {
using common::BBox;
using common::InferResult;
using common::Points;

using common::ClsRet;
using common::DetRet;
using common::PoseRet;

/**
 * @brief 模型的基础信息（模型装载后可以获取）
 *
 */
struct ModelInfo {
  int output_count;                           // 输出的个数
  std::vector<std::vector<int>> outputShapes; // 输出的尺度
};

} // namespace infer

#endif