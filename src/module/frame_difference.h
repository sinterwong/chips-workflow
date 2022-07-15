#ifndef __SOLUTION_FRAME_DIFFERENCE_H_
#define __SOLUTION_FRAME_DIFFERENCE_H_

///运动物体检测――帧差法
#include "opencv2/opencv.hpp"
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace solution {

class FrameDifference {
  cv::Mat result;
  std::shared_ptr<cv::Mat> lastFrame;

  bool
  moveDetect(const cv::Mat &temp, const cv::Mat &frame,
             std::vector<std::pair<std::string, std::array<float, 5>>> &bboxes);

public:
  FrameDifference() = default;
  FrameDifference(std::shared_ptr<cv::Mat> &frame) : lastFrame(frame){};
  void init(std::shared_ptr<cv::Mat> &frame) { lastFrame = frame; };
  bool
  update(std::shared_ptr<cv::Mat> &frame,
         std::vector<std::pair<std::string, std::array<float, 5>>> &bboxes);
  inline bool statue() { return !lastFrame; }
};
} // namespace solution
#endif