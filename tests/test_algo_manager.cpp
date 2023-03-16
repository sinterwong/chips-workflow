#include <gflags/gflags.h>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <vector>

#include "logger/logger.hpp"
#include "messageBus.h"
#include "preprocess.hpp"

#include "infer/algorithmManager.hpp"

DEFINE_string(image_path, "", "Specify image path.");
DEFINE_string(model_path, "", "Specify the yolo model path.");

using common::RetBox;
using common::AlgoBase;
using common::DetAlgo;
using common::InferParams;
using common::InferResult;
using common::Shape;
using infer::AlgorithmManager;

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

  Shape inputShape{640, 640, 3};
  AlgoBase base_config{
      1,
      std::move(inputNames),
      std::move(outputNames),
      FLAGS_model_path,
      "Yolo",
      std::move(inputShape),
      false,
      alpha,
      0,
  };

  DetAlgo det_config{std::move(base_config), 0.3, 0.4};

  AlgoConfig config;
  config.setParams(std::move(det_config));

  RetBox region{"hello", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

  InferParams params{std::string("hello"),
                     ColorType::NV12,
                     0.5,
                     region,
                     {image_nv12.cols, image_nv12.rows, image_nv12.channels()}};
  InferResult ret;

  std::unique_ptr<AlgorithmBus> algo_bus;

  algo_bus = std::make_unique<AlgorithmManager>();

  algo_bus->registered("handdet_yolov5", config);

  algo_bus->infer("handdet_yolov5_dummy", image_nv12.data, params, ret);

  algo_bus->infer("handdet_yolov5", image_nv12.data, params, ret);

  algo_bus->unregistered("handdet_yolov5");

  auto bboxes = std::get_if<common::BBoxes>(&ret.aRet);
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
  cv::imwrite("test_vision_infer_out.jpg", image_bgr);

  gflags::ShutDownCommandLineFlags();
  return 0;
}
