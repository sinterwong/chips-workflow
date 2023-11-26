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
#include "facelib.hpp"
#include "logger/logger.hpp"
#include <gflags/gflags.h>
#include <opencv2/imgcodecs.hpp>

DEFINE_string(img1, "", "Specify a image path which contains some face.");
DEFINE_string(img2, "", "Specify a image path which contains some face.");

using namespace server::face;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

void getFrameInput(cv::Mat &input, FrameInfo &frame) {
  frame.data = reinterpret_cast<void **>(&input.data);
  frame.inputShape = {input.cols, input.rows, input.channels()};

  // 暂时写死NV12格式，这里应该有一个宏来确定是什么推理数据
  frame.shape = {input.cols, input.rows * 2 / 3, input.channels()};
  frame.type = common::ColorType::NV12;
}

int main(int argc, char **argv) {

  FlowEngineLoggerSetLevel(1);
  gflags::SetUsageMessage("Face recognition");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  core::FaceLibrary facelib(512);

  // cv::Mat im1 = cv::imread(FLAGS_img1);
  // cv::Mat im2 = cv::imread(FLAGS_img2);
  // cv::Mat im1_input, im2_input;
  // infer::utils::BGR2NV12(im1, im1_input);
  // infer::utils::BGR2NV12(im2, im2_input);
  // FrameInfo frame1, frame2;
  // getFrameInput(im1_input, frame1);
  // getFrameInput(im2_input, frame2);

  std::vector<float> f1, f2;

  if (!core::FaceRecognition::getInstance().extract(FLAGS_img1, f1)) {
    return -1;
  }
  if (!core::FaceRecognition::getInstance().extract(FLAGS_img2, f2)) {
    return -1;
  }

  facelib.addVector(f1.data(), 1);

  auto ret = facelib.search(f2.data(), 1).at(0);
  std::cout << "ID: " << ret.first << ", Distance: " << ret.second << std::endl;

  gflags::ShutDownCommandLineFlags();
  return 0;
}