#include <gflags/gflags.h>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>

#include "logger/logger.hpp"
#include "messageBus.h"
#include "preprocess.hpp"
#include "visionInfer.hpp"

DEFINE_string(image_path, "", "Specify image path.");
DEFINE_string(model_path, "", "Specify the yolo model path.");

using common::InferResult;
using infer::VisionInfer;

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cv::Mat image_bgr = cv::imread(FLAGS_image_path);
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  float alpha = 0;
  inputNames = {"images"};
  outputNames = {"output"};
  alpha = 255.0;
  cv::Mat image_rgb;
  cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

  cv::Mat image_nv12;
  infer::utils::RGB2NV12(image_rgb, image_nv12);
  std::shared_ptr<cv::Mat> image_nv12_ptr =
      std::make_shared<cv::Mat>(image_nv12);

  std::array<int, 3> inputShape{640, 640, 3};
  common::AlgorithmConfig config{FLAGS_model_path,
                                 std::move(inputNames),
                                 std::move(outputNames),
                                 std::move(inputShape),
                                 "Yolo",
                                 0.3,
                                 0.4,
                                 alpha,
                                 0,
                                 false,
                                 1};

  std::shared_ptr<AlgoInfer> vision = std::make_shared<VisionInfer>(config);
  if (!vision->init()) {
    FLOWENGINE_LOGGER_ERROR("Failed to init vison");
    return -1;
  }

  std::vector<RetBox> regions{{"hello", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};

  InferParams params{
      std::string("hello"), ColorType::NV12, 0.5, regions, {0} // [0, 2, 10...]
  };
  InferResult ret;

  vision->infer(&image_nv12, params, ret);

  auto bboxes = std::get_if<common::DetRet>(&ret.aRet);
  if (!bboxes) {
    FLOWENGINE_LOGGER_ERROR("Wrong algorithm type!");
    return -1;
  }

  FLOWENGINE_LOGGER_INFO("number of result: {}", bboxes->size());
  for (auto &bbox : *bboxes) {
    cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
                  bbox.bbox[3] - bbox.bbox[1]);
    cv::rectangle(image_bgr, rect, cv::Scalar(0, 0, 255), 2);
  }
  cv::cvtColor(image_bgr, image_bgr, cv::COLOR_RGB2BGR);
  cv::imwrite("test_det_out.jpg", image_bgr);

  gflags::ShutDownCommandLineFlags();
  return 0;
}
