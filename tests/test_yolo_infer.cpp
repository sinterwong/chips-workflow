#include "common/common.hpp"
#include "gflags/gflags.h"
#include "hb_comm_video.h"
#include "logger/logger.hpp"
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "yoloDet.hpp"
#include "infer_utils.hpp"
#if TARGET_PLATFORM == x3
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#elif TARGET_PLATFORM == jetson
#include "jetson/trt_inference.hpp"
using namespace infer::jetson;
#endif

using namespace infer;
DEFINE_string(image_path, "", "Specify image path.");
DEFINE_string(model_path, "", "Specify model path.");
DEFINE_int32(input_height, 640, "Specify input height.");
DEFINE_int32(input_width, 640, "Specify input width.");

int main(int argc, char **argv) {
  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::array<int, 3> inputShape{FLAGS_input_width, FLAGS_input_height, 3};

  std::vector<std::string> inputNames = {"images"};
  std::vector<std::string> outputNames = {"output"};
  common::AlgorithmConfig params{FLAGS_model_path,
                                 std::move(inputNames),
                                 std::move(outputNames),
                                 std::move(inputShape),
                                 "yolo",
                                 0.4,
                                 0.4,
                                 255.0,
                                 0,
                                 false,
                                 1};
  
  std::shared_ptr<AlgoInference> instance = std::make_shared<AlgoInference>(params);
  if (!instance->initialize()) {
    FLOWENGINE_LOGGER_ERROR("YoloDet initialization is failed!");
    return -1;
  }

  ModelInfo info;
  instance->getModelInfo(info);
  std::shared_ptr<vision::YoloDet> det = std::make_shared<vision::YoloDet>(params, info);

  VIDEO_FRAME_S frameInfo;

  cv::Mat image = cv::imread(FLAGS_image_path);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  
  // cv::Mat yv12;
  // cv::cvtColor(image, yv12, cv::COLOR_RGB2YUV_YV12);
  // cv::Mat temp;
  // cv::cvtColor(yv12, temp, CV_YUV2BGR_YV12);
  // cv::imwrite("temp.jpg", temp); // only for test

  cv::Mat nv12;
  utils::RGB2NV12(image, nv12);
  cv::Mat picBGR;
  cv::cvtColor(nv12, picBGR, CV_YUV2BGR_NV12);
  cv::imwrite("test.jpg", picBGR); // only for test
  // FLOWENGINE_LOGGER_INFO("Saved the test image");

  frameInfo.stVFrame.height = image.rows;
  frameInfo.stVFrame.width = image.cols;
  frameInfo.stVFrame.vir_ptr[0] = reinterpret_cast<hb_char*>(nv12.data);
  
  void* output = nullptr;
  FrameInfo input;
  input.data = reinterpret_cast<void**>(frameInfo.stVFrame.vir_ptr);
  input.shape = {image.cols, image.rows, 3};
  
  if (!instance->infer(input, &output)) {
    FLOWENGINE_LOGGER_ERROR("infer is failed!");
    return -1;
  }

  infer::Result result;
  result.shape = {image.cols, image.rows, 3};
  det->processOutput(output, result);

  FLOWENGINE_LOGGER_INFO("number of result: {}", result.detResults.size());
  for (auto &bbox : result.detResults) {
    cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
                  bbox.bbox[3] - bbox.bbox[1]);
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("test_yolo_out.jpg", image);

  gflags::ShutDownCommandLineFlags();
  // FlowEngineLoggerDrop();

  return 0;
}