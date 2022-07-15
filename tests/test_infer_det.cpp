/**
 * @file jetson_det.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-06-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <opencv2/opencv.hpp>
#include <thread>

// #include "jetson/jetsonOutputModule.h"
// #include "jetson/jetsonSourceModule.h"
// #include "boostMessage.h"
#include "jetson/detection.hpp"
#include "jetson/trt_inference.hpp"

cv::Rect get_rect(cv::Mat &img, int width, int height,
                  std::array<float, 4> bbox) {
  int l, r, t, b;
  float r_w = width / (img.cols * 1.0);
  float r_h = height / (img.rows * 1.0);
  if (r_h > r_w) {
    l = bbox[0] - bbox[2] / 2.f;
    r = bbox[0] + bbox[2] / 2.f;
    t = bbox[1] - bbox[3] / 2.f - (height - r_w * img.rows) / 2;
    b = bbox[1] + bbox[3] / 2.f - (height - r_w * img.rows) / 2;
    l = l / r_w;
    r = r / r_w;
    t = t / r_w;
    b = b / r_w;
  } else {
    l = bbox[0] - bbox[2] / 2.f - (width - r_h * img.cols) / 2;
    r = bbox[0] + bbox[2] / 2.f - (width - r_h * img.cols) / 2;
    t = bbox[1] - bbox[3] / 2.f;
    b = bbox[1] + bbox[3] / 2.f;
    l = l / r_h;
    r = r / r_h;
    t = t / r_h;
    b = b / r_h;
  }
  return cv::Rect(l, t, r - l, b - t);
}

int main() {
  infer::InferParams params;
  params.batchSize = 1;
  params.numAnchors = 25200;
  params.numClasses = 80;
  params.serializedFilePath = "/home/wangxt/workspace/projects/flowengine/tests/data/yolov5s.engine";
  params.inputTensorNames.push_back("images");
  params.outputTensorNames.push_back("output");
  params.inputShape = {640, 640, 3};
  params.originShape = {1920, 1080, 3};
  params.scaling = float(params.inputShape[0]) /
                   (float)std::max(params.originShape[0], params.originShape[1]);
  std::shared_ptr<infer::trt::DetctionInfer> instance(
      std::make_shared<infer::trt::DetctionInfer>(std::move(params)));
  instance->initialize();
  infer::Result output;
  cv::Mat frame = cv::imread("/home/wangxt/workspace/projects/flowengine/tests/data/traffic.jpg");
  // if (frame.empty()) {
  //   std::cout << "Test image is empty!" << std::endl;
  //   exit(-1);
  // }
  
  // std::vector<cv::Mat> inputs;
  // inputs.push_back(frame);
  instance->infer(frame.data, output);
  std::cout << "Object number: " << output.detResults[0].size() << std::endl;
  for (size_t j = 0; j < output.detResults[0].size(); j++) {
      cv::Rect r = get_rect(frame, 640, 640, output.detResults[0][j].bbox);
      cv::rectangle(frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
      cv::putText(frame,
      std::to_string((int)output.detResults[0][j].class_id),
                  cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                  cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    }
    cv::imwrite("_out.jpg", frame);

//   BoostMessage bus;
//   Backend backend(&bus);
//   // rtsp://user:passward@114.242.23.39:9201/test
//   // /home/wangxt/workspace/projects/flowengine/sample_1080p_h264.mp4
//   std::shared_ptr<JetsonSourceModule> cap(new JetsonSourceModule(
//       &backend,
//       "/home/wangxt/workspace/projects/flowengine/sample_1080p_h264.mp4", 1920,
//       1080, "h264", "Camera", "FrameMessage", {}, {"WriteVideo"}));

//   std::shared_ptr<JetsonOutputModule> output(new JetsonOutputModule(
//       &backend, "out.mp4", "WriteVideo", "FrameMessage", {"Camera"}));

  // std::thread th1(&JetsonOutputModule::go, cap);
  // std::thread th2(&JetsonOutputModule::go, output);
  // th1.join();
  // th2.join();

  return 0;
}