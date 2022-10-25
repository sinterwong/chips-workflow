#include "inference.h"
#include "jetson/assdDet.hpp"
#include "jetson/detection.hpp"
#include "jetson/yoloDet.hpp"
#include <chrono>
#include <cstdint>
#include <gflags/gflags.h>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

DEFINE_string(image_path, "", "Specify config path.");
DEFINE_string(model_path, "", "Specify model path.");
DEFINE_int32(input_height, 335, "Specify input height.");
DEFINE_int32(input_width, 335, "Specify input width.");

using namespace infer::trt;

int main(int argc, char **argv) {

  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> inputNames = {"data"};
  // std::vector<std::string> outputNames = {"output2_conv3", "output3_conv3",
  //                                         "output4_conv3"};
  std::vector<std::string> outputNames = {"conv13_3", "conv15_3",
                                          "conv16_3"};

  std::array<int, 3> inputShape{FLAGS_input_width, FLAGS_input_height, 3};
  common::AlgorithmConfig params{FLAGS_model_path,
                                 std::move(inputNames),
                                 std::move(outputNames),
                                 std::move(inputShape),
                                 "assd",
                                 0.4,
                                 0.4,
                                 0,
                                 0,
                                 false,
                                 1};
  std::shared_ptr<DetectionInfer> det = std::make_shared<AssdDet>(params);
  det->initialize();

  cv::Mat image = cv::imread(FLAGS_image_path);

  infer::Result result;
  result.shape = {image.cols, image.rows, 3};
  det->infer(image.data, result);
  FLOWENGINE_LOGGER_INFO("number of result: {}", result.detResults.size());
  for (auto &bbox : result.detResults) {
    cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
                  bbox.bbox[3] - bbox.bbox[1]);
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("test_assd_out.jpg", image);

  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}

/*
./test_assd
--image_path=../../../tests/data/pedestrian.jpg
--model_path=../../../tests/data/model/assd_shoulder.engine --input_height=543
--input_width=967
*/
