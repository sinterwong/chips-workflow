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
using common::Point;
using common::Shape;

using common::InferResult;

using common::ClsRet;
using common::DetRet;
using common::PoseRet;
using common::ModelInfo;

} // namespace infer

#endif