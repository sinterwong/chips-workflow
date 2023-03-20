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
using common::Point2f;
using common::KeypointsBox;
using common::KeypointsBoxes;
using common::Shape;

using common::InferResult;

using common::ClsRet;
using common::BBoxes;
using common::Points2f;
using common::ModelInfo;

using common::AlgoConfig;
using common::AlgoBase;
using common::ClassAlgo;
using common::DetAlgo;

} // namespace infer

#endif