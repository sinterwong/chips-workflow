/**
 * @file frame_difference.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-15
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __SOLUTION_FRAME_DIFFERENCE_H_
#define __SOLUTION_FRAME_DIFFERENCE_H_

/// 运动物体检测――帧差法
#include "opencv2/opencv.hpp"
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace infer::solution {

using RetBox = std::pair<std::string, std::array<float, 6>>;

class FrameDifference {
  // std::shared_ptr<cv::Mat> lastFrame;
  std::shared_ptr<cv::Mat> lastFrameProcessed;

public:
  FrameDifference(){};
  bool update(cv::Mat &frame, std::vector<RetBox> &bboxes);
};
} // namespace infer::solution
#endif