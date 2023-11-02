/**
 * @file test_face_rec.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-02
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "faceRecognition.hpp"
#include "logger/logger.hpp"
#include <opencv2/imgcodecs.hpp>

#include <gflags/gflags.h>

DEFINE_string(img, "", "Specify a image which contains some face path.");

using namespace server::face;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

int main(int argc, char **argv) {

  FlowEngineLoggerSetLevel(1);
  gflags::SetUsageMessage("Face recognition");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  core::FaceRecognition faceRec;
  cv::Mat image = cv::imread(FLAGS_img);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  std::vector<float> feature;
  faceRec.forward(image, feature);

  for (auto f : feature) {
    std::cout << f << ", ";
  }
  std::cout << std::endl;
  gflags::ShutDownCommandLineFlags();

  return 0;
}