#include "assdDet.hpp"
#include "detection.hpp"
#include "inference.h"
#include "yoloDet.hpp"
#include <chrono>
#include <cstdint>
#include <gflags/gflags.h>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

#if (TARGET_PLATFORM == 0)
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#elif (TARGET_PLATFORM == 1)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
#endif
using namespace infer::vision;

DEFINE_string(image_path, "", "Specify config path.");
DEFINE_string(model_path, "", "Specify model path.");
DEFINE_int32(input_height, 335, "Specify input height.");
DEFINE_int32(input_width, 335, "Specify input width.");

int main(int argc, char **argv) {

  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> inputNames = {"data"};
  // std::vector<std::string> outputNames = {"output2_conv3", "output3_conv3",
  //                                         "output4_conv3"};
  std::vector<std::string> outputNames = {"conv13_3", "conv15_3", "conv16_3"};

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
  std::shared_ptr<AlgoInference> instance =
      std::make_shared<AlgoInference>(params);
  if (!instance->initialize()) {
    FLOWENGINE_LOGGER_ERROR("YoloDet initialization is failed!");
    return -1;
  }
  infer::ModelInfo info;
  instance->getModelInfo(info);
  std::shared_ptr<YoloDet> det = std::make_shared<YoloDet>(params, info);

  // TODO 根据不同平台，给出不同的类型
  cv::Mat image = cv::imread(FLAGS_image_path);

  infer::Result ret;
  ret.shape = {image.cols, image.rows, 3};
  infer::FrameInfo frame;
  frame.data = reinterpret_cast<void **>(&image.data);
  frame.shape = ret.shape;
  void *output = nullptr;
  instance->infer(frame, &output);
  det->processOutput(output, ret);
  FLOWENGINE_LOGGER_INFO("number of result: {}", ret.detResults.size());
  for (auto &bbox : ret.detResults) {
    cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
                  bbox.bbox[3] - bbox.bbox[1]);
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("test_det_out.jpg", image);

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
