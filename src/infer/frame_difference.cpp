/**
 * @file frame_difference.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-15
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "frame_difference.h"
#include "logger/logger.hpp"
#include <future>
#include <opencv2/imgcodecs.hpp>
#include <utility>

namespace infer::solution {

void FrameDifference::preprocessing(cv::Mat &input, cv::Mat &output) {
  // 平滑、帧差或背景差、二值化、膨胀、腐蚀。
  // 平滑处理
  cv::blur(input, output, cv::Size(3, 3), cv::Point(-1, -1)); // filter2D

  // 将background和frame转为灰度图
  cv::Mat current_gray;
  cv::cvtColor(output, output, cv::COLOR_RGB2GRAY);
}

void FrameDifference::frameDiff(cv::Mat &previous, cv::Mat &current,
                                cv::Mat &output) {
  // 将background和frame做差
  cv::Mat diff;
  cv::absdiff(previous, current, diff);
  /*
  getStructuringElement()
  这个函数的第一个参数表示内核的形状，有三种形状可以选择。
  矩形：MORPH_RECT;
  交叉形：MORPH_CORSS;
  椭圆形：MORPH_ELLIPSE;
  第二和第三个参数分别是内核的尺寸以及锚点的位置
  */
  cv::Mat kernel_erode = getStructuringElement(
      cv::MORPH_RECT, cv::Size(3, 3)); // 函数会返回指定形状和尺寸的结构元素。
  // 调用之后，调用膨胀与腐蚀函数的时候，第三个参数值保存了getStructuringElement返回值的Mat类型变量。也就是element变量。
  cv::Mat kernel_dilate =
      getStructuringElement(cv::MORPH_RECT, cv::Size(18, 18));

  // 进行二值化处理，选择50，255为阈值
  cv::threshold(diff, output, 50, 255, cv::THRESH_BINARY);
  // cv::imwrite("threshold.jpg", diff_thresh);
  // 膨胀
  cv::dilate(output, output, kernel_dilate);
  // cv::imwrite("dilate.jpg", diff_thresh);
  // 腐蚀
  cv::erode(output, output, kernel_erode);
  // cv::imwrite("erode.jpg", diff_thresh);
}

bool FrameDifference::init(cv::Mat &frame) {
  cv::Mat firstGray;
  preprocessing(frame, firstGray);
  previousFrameProcessed = std::make_shared<cv::Mat>(firstGray);
  return true;
}

bool FrameDifference::update(cv::Mat &frame, std::vector<RetBox> &bboxes,
                             float threshold) {
  if (frame.empty()) {
    return false;
  }

  cv::Mat current_gray;
  preprocessing(frame, current_gray);
  // 判断是否是第一帧
  if (!previousFrameProcessed) {
    previousFrameProcessed = std::make_shared<cv::Mat>(current_gray);
    FLOWENGINE_LOGGER_INFO("FrameDifference first frame");
    return true;
  }

  // 帧差后的二值图结果
  cv::Mat diff_thresh;

  // 帧差
  frameDiff(*previousFrameProcessed, current_gray, diff_thresh);

  // 更新上一帧
  previousFrameProcessed = std::make_shared<cv::Mat>(current_gray);

  // 查找轮廓并绘制轮廓
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(diff_thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE); // 找轮廓函数
  // 绘制轮廓
  // cv::drawContours(frame, contours, -1, cv::Scalar(255, 78, 86), 2);
  // 查找正外接矩形
  for (size_t i = 0; i < contours.size(); i++) {
    cv::Rect rect = cv::boundingRect(contours[i]);
    // 如果边界矩形的面积小于图像的某个比例，则认为是噪声。
    if (rect.width * rect.height < threshold * frame.cols * frame.rows) {
      continue;
    }

    RetBox bbox{"FrameDifference", rect.x, rect.y, rect.width, rect.height};
    bboxes.emplace_back(std::move(bbox));
  }
  return true;
}
} // namespace infer::solution
