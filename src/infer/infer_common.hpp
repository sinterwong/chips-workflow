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
using Point2f = common::Point<float>;
using common::KeypointsBox;
using common::KeypointsBoxes;
using common::Shape;

using common::InferResult;

using common::BBoxes;
using common::CharsRet;
using common::ClsRet;
using common::Eigenvector;
using common::ModelInfo;
using common::Points2f;

using common::AlgoBase;
using common::AlgoConfig;
using common::ClassAlgo;
using common::DetAlgo;
using common::FeatureAlgo;
using common::PointsDetAlgo;

} // namespace infer

#endif