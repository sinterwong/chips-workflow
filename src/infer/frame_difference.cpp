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
#include <opencv2/imgcodecs.hpp>
#include <utility>

namespace solution {

bool FrameDifference::update(std::shared_ptr<cv::Mat> &frame,
                             std::vector<RetBox> &bboxes) {
  if (!lastFrame) {
    return false;
  }
  //处理帧
  if (!moveDetect(*lastFrame, *frame, bboxes)) {
    return false;
  };

  lastFrame = frame;
  return true;
}

//运动物体检测函数声明
bool FrameDifference::moveDetect(const cv::Mat &temp, const cv::Mat &frame,
                                 std::vector<RetBox> &bboxes) {

  //平滑、帧差或背景差、二值化、膨胀、腐蚀。
  // 1.平滑处理
  cv::Mat dst_temp;
  cv::blur(temp, dst_temp, cv::Size(3, 3), cv::Point(-1, -1)); // filter2D

  cv::Mat dst_frame;
  cv::blur(frame, dst_frame, cv::Size(3, 3), cv::Point(-1, -1)); // filter2D

  // 2.帧差
  // 2.1将background和frame转为灰度图
  cv::Mat gray1, gray2;
  cv::cvtColor(dst_temp, gray1, cv::COLOR_BGR2GRAY);
  cv::cvtColor(dst_frame, gray2, cv::COLOR_BGR2GRAY);
  // 2.2.将background和frame做差
  cv::Mat diff;
  cv::absdiff(gray1, gray2, diff);
  // cv::imshow("absdiff", diff);

  // 3.对差值图diff_thresh进行阈值化处理  二值化
  cv::Mat diff_thresh;

  /*
  getStructuringElement()
  这个函数的第一个参数表示内核的形状，有三种形状可以选择。
  矩形：MORPH_RECT;
  交叉形：MORPH_CORSS;
  椭圆形：MORPH_ELLIPSE;
  第二和第三个参数分别是内核的尺寸以及锚点的位置
  */
  cv::Mat kernel_erode = getStructuringElement(
      cv::MORPH_RECT, cv::Size(3, 3)); //函数会返回指定形状和尺寸的结构元素。
  //调用之后，调用膨胀与腐蚀函数的时候，第三个参数值保存了getStructuringElement返回值的Mat类型变量。也就是element变量。
  cv::Mat kernel_dilate =
      getStructuringElement(cv::MORPH_RECT, cv::Size(18, 18));

  //进行二值化处理，选择50，255为阈值
  cv::threshold(diff, diff_thresh, 50, 255, cv::THRESH_BINARY);
  // cv::imshow("threshold", diff_thresh);
  // 4.膨胀
  cv::dilate(diff_thresh, diff_thresh, kernel_dilate);
  // cv::imshow("dilate", diff_thresh);
  // 5.腐蚀
  cv::erode(diff_thresh, diff_thresh, kernel_erode);
  // cv::imshow("erode", diff_thresh);

  // 6.查找轮廓并绘制轮廓
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(diff_thresh, contours, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE); //找轮廓函数
  cv::drawContours(result, contours, -1, cv::Scalar(0, 0, 255),
                   2); //在result上绘制轮廓
  // 7.查找正外接矩形
  std::vector<cv::Rect> boundRect(contours.size());
  for (size_t i = 0; i < contours.size(); i++) {
    boundRect[i] = cv::boundingRect(contours[i]);
    RetBox bbox = {
        name,
        {static_cast<float>(boundRect[i].x), static_cast<float>(boundRect[i].y),
         static_cast<float>(boundRect[i].x + boundRect[i].width),
         static_cast<float>(boundRect[i].y + boundRect[i].height), 0.0, 0.0}};
    bboxes.emplace_back(std::move(bbox));
  }
  return true;
}
} // namespace solution
