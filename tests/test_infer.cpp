#include "common/common.hpp"
#include "gflags/gflags.h"
#include "infer_utils.hpp"
#include "logger/logger.hpp"
#include "yoloDet.hpp"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#if (TARGET_PLATFORM == 0)
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#include "hb_comm_video.h"
#include "hb_type.h"
#elif (TARGET_PLATFORM == 1)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
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
                                 "Yolo",
                                 0.4,
                                 0.4,
                                 255.0,
                                 0,
                                 false,
                                 1};

  std::shared_ptr<AlgoInference> instance =
      std::make_shared<AlgoInference>(params);
  if (!instance->initialize()) {
    FLOWENGINE_LOGGER_ERROR("Yolo initialization is failed!");
    return -1;
  }

  ModelInfo info;
  instance->getModelInfo(info);
  std::shared_ptr<vision::Yolo> det =
      std::make_shared<vision::Yolo>(params, info);
  /*
  VIDEO_FRAME_S frameInfo;
  memset(&frameInfo, 0, sizeof(VIDEO_FRAME_S));

  cv::Mat image = cv::imread(FLAGS_image_path);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  cv::Mat nv12;
  utils::RGB2NV12(image, nv12);
  // std::cout << "nv12.cols: " << nv12.cols << ", "
  //           << "nv12.rows: " << nv12.rows << std::endl;
  // cv::imwrite("nv12.png", nv12);
  // std::cout << "image.cols: " << image.cols << ", "
  //           << "image.rows: " << image.rows << std::endl;
  // cv::Mat y = nv12(cv::Rect(0, 0, image.cols, image.rows)).clone();
  // cv::Mat uv =
  //     nv12(cv::Rect(0, image.rows, nv12.cols, nv12.rows - image.rows)).clone();
  // cv::imwrite("y.png", y);
  // cv::imwrite("uv.png", uv);

  frameInfo.stVFrame.height = image.rows;
  frameInfo.stVFrame.width = image.cols;
  frameInfo.stVFrame.size = image.rows * image.cols * 3 / 2;
  frameInfo.stVFrame.vir_ptr[0] = reinterpret_cast<hb_char *>(nv12.data);
  frameInfo.stVFrame.vir_ptr[1] =
      reinterpret_cast<hb_char *>(nv12.data) + (image.rows * image.cols);
  cv::Mat picNV12 = cv::Mat(image.rows * 3 / 2, image.cols, CV_8UC1,
                            frameInfo.stVFrame.vir_ptr[0]);
  cv::Mat temp_y = cv::Mat(image.rows, image.cols, CV_8UC1,
                            frameInfo.stVFrame.vir_ptr[0]);
  cv::Mat temp_uv = cv::Mat(image.rows / 2, image.cols, CV_8UC1,
                            frameInfo.stVFrame.vir_ptr[1]);
  cv::imwrite("nv12_finial.png", picNV12);
  cv::imwrite("y_finial.png", temp_y);
  cv::imwrite("uv_finial.png", temp_uv);

  // for (int i = 0; i < image.rows * image.cols; i ++) {
  //   std::cout << static_cast<int>(frameInfo.stVFrame.vir_ptr[0][i]) << ", ";
  // }
  // std::cout << std::endl;

  void *output = nullptr;
  FrameInfo input;
  input.data = reinterpret_cast<void **>(frameInfo.stVFrame.vir_ptr);
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
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
  cv::imwrite("test_yolo_out.jpg", image);
  */

  gflags::ShutDownCommandLineFlags();
  // FlowEngineLoggerDrop();

  return 0;
}