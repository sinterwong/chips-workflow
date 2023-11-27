/**
 * @file postprocess.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __INFERENCE_UTILS_POSTPROCESS_H_
#define __INFERENCE_UTILS_POSTPROCESS_H_
#include "infer_common.hpp"
#include "opencv2/imgproc.hpp"

namespace infer::utils {

float iou(std::array<float, 4> const &, std::array<float, 4> const &);

void nms(BBoxes &, std::unordered_map<int, BBoxes> &, float);

void nms_kbox(KeypointsBoxes &, std::unordered_map<int, KeypointsBoxes> &,
              float);

// 复原points原始尺寸
void restoryPoints(Points2f &results, float rw, float rh);

// 复原bbox原始尺寸
void restoryBox(BBox &ret, float rw, float rh, Shape const &shape);

void restoryBoxes(BBoxes &results, Shape const &shape, Shape const &inputShape,
                  bool isScale);

void restoryKeypointsBoxes(KeypointsBoxes &results, Shape const &shape,
                           Shape const &inputShape, bool isScale);

// 四个点的仿射变换
void fourPointTransform(cv::Mat &input, cv::Mat &output,
                        infer::Points2f const &points);

// 按照左上、右上、右下、左下的顺序排序
void sortFourPoints(Points2f &points);

// 获取最中心的bbox
size_t findClosestBBoxIndex(KeypointsBoxes const &kbboxes, float w, float h);
} // namespace infer::utils
#endif