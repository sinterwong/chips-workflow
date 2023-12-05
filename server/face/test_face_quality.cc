/**
 * @file test_face_quality.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "faceQuality.hpp"
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>

DEFINE_string(img, "", "Specify a image path which contains some face.");

using namespace server::face::core;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

int main(int argc, char **argv) {
  FlowEngineLoggerSetLevel(1);
  gflags::SetUsageMessage("Face recognition");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  FaceQuality &faceQuality = FaceQuality::getInstance();

  cv::Mat image_bgr = cv::imread(FLAGS_img);

  cv::Mat image_nv12;
  infer::utils::BGR2NV12(image_bgr, image_nv12);

  FramePackage framePackage;
  framePackage.frame = std::make_shared<cv::Mat>(image_nv12);

  int quality = -1;
  faceQuality.infer(framePackage, quality);

  std::cout << "quality: " << quality << std::endl;

  gflags::ShutDownCommandLineFlags();
  return 0;
}