/**
 * @file faceLiveness.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "faceLiveness.hpp"
#include "algoManager.hpp"
#include "logger/logger.hpp"
#include "postprocess.hpp"

using namespace infer;

using namespace common;

namespace server::face::core {

FaceLiveness *FaceLiveness::instance = nullptr;

bool FaceLiveness::infer(FramePackage const &framePackage, int &liveness) {

  // 使用算法资源进行推理
  FrameInfo frame;
  frame.data = reinterpret_cast<void **>(&framePackage.frame->data);
  frame.inputShape = {framePackage.frame->cols, framePackage.frame->rows,
                      framePackage.frame->channels()};

  return true;
}

cv::Mat FaceLiveness::transform(int size, cv::Point2f center, double scale,
                                double rotation) {
  // Step 1: 缩放变换
  cv::Mat scaleMat = cv::getRotationMatrix2D(center, 0, scale);
  cv::Mat scaleMat3x3 = cv::Mat::eye(3, 3, scaleMat.type());
  scaleMat.copyTo(scaleMat3x3.rowRange(0, 2));

  // Step 2: 向原始中心的负方向平移
  cv::Mat transToNegCenterMat = cv::Mat::eye(3, 3, scaleMat.type());
  transToNegCenterMat.at<double>(0, 2) = -center.x;
  transToNegCenterMat.at<double>(1, 2) = -center.y;

  // Step 3: 旋转变换
  cv::Mat rotMat = cv::getRotationMatrix2D(cv::Point2f(0, 0), rotation, 1);
  cv::Mat rotMat3x3 = cv::Mat::eye(3, 3, rotMat.type());
  rotMat.copyTo(rotMat3x3.rowRange(0, 2));

  // Step 4: 向图像中心的平移
  cv::Mat transToImgCenterMat = cv::Mat::eye(3, 3, rotMat.type());
  transToImgCenterMat.at<double>(0, 2) = size / 2.0;
  transToImgCenterMat.at<double>(1, 2) = size / 2.0;

  // 组合所有变换
  cv::Mat combinedMat3x3 =
      transToImgCenterMat * rotMat3x3 * transToNegCenterMat * scaleMat3x3;

  // 将3x3矩阵转换回2x3矩阵
  cv::Mat combinedMat = combinedMat3x3.rowRange(0, 2);

  combinedMat.convertTo(combinedMat, CV_32FC1);

  return combinedMat;
}

cv::Mat FaceLiveness::normCrop(cv::Mat const &image, cv::Mat const &rotMat,
                               int size) {
  cv::Mat aligned_face;
  cv::warpAffine(image, aligned_face, rotMat, cv::Size(size, size));
  return aligned_face;
}

void FaceLiveness::getFaceInput(cv::Mat const &input, cv::Mat &output,
                                FrameInfo &frame, Points2f const &points,
                                ColorType const &type) {}

} // namespace server::face::core