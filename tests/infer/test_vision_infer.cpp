#include <gflags/gflags.h>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <utility>
#include <variant>
#include <vector>

#include "logger/logger.hpp"
#include "messageBus.h"
#include "preprocess.hpp"
#include "visionInfer.hpp"

DEFINE_string(image_path, "", "Specify image path.");
DEFINE_string(model_path, "", "Specify the yolo model path.");

using common::AlgoBase;
using common::DetAlgo;
using common::ClassAlgo;
using common::InferResult;
using common::RetBox;
using common::Shape;
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

  Shape inputShape{176, 48, 3};
  AlgoBase base_config{
      1,
      std::move(inputNames),
      std::move(outputNames),
      FLAGS_model_path,
      "CRNN",
      std::move(inputShape),
      false,
      alpha,
      0,
      0.3,
  };

  // DetAlgo det_config{std::move(base_config), 0.4};
  ClassAlgo cls_config{std::move(base_config)};

  AlgoConfig center;

  // center.setParams(det_config);
  center.setParams(cls_config);

  std::shared_ptr<AlgoInfer> vision = std::make_shared<VisionInfer>(center);
  if (!vision->init()) {
    FLOWENGINE_LOGGER_ERROR("Failed to init vision");
    return -1;
  }

  RetBox region{"hello"};

  InferParams params{std::string("hello"),
                     ColorType::NV12,
                     0.0,
                     region,
                     {image_nv12.cols, image_nv12.rows, image_nv12.channels()}};
  InferResult ret;

  // 制作输入数据
  FrameInfo frame;
  frame.shape = {image_nv12.cols, image_nv12.rows * 2 / 3, 3};
  frame.type = params.frameType;
  frame.data = reinterpret_cast<void **>(&image_nv12.data);
  vision->infer(frame, params, ret);

  // auto bboxes = std::get_if<common::BBoxes>(&ret.aRet);
  // auto bboxes = std::get_if<common::KeypointsBoxes>(&ret.aRet);
  auto chars = std::get_if<common::CharsRet>(&ret.aRet);

  // if (!bboxes) {
  //   FLOWENGINE_LOGGER_ERROR("Wrong algorithm type!");
  //   return -1;
  // }

  if (!chars) {
    FLOWENGINE_LOGGER_ERROR("Wrong algorithm type!");
    return -1;
  }

  // FLOWENGINE_LOGGER_INFO("number of result: {}", bboxes->size());
  // for (auto &bbox : *bboxes) {
  //   // cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
  //   //               bbox.bbox[3] - bbox.bbox[1]);
  //   // cv::rectangle(image_bgr, rect, cv::Scalar(0, 0, 255), 2);

  //   cv::Rect rect(bbox.bbox.bbox[0], bbox.bbox.bbox[1],
  //                 bbox.bbox.bbox[2] - bbox.bbox.bbox[0],
  //                 bbox.bbox.bbox[3] - bbox.bbox.bbox[1]);
  //   cv::rectangle(image_bgr, rect, cv::Scalar(0, 0, 255), 2);
  //   for (auto &p : bbox.points) {
  //     cv::circle(
  //         image_bgr,
  //         cv::Point{static_cast<int>(p.at(0)), static_cast<int>(p.at(1))}, 3,
  //         cv::Scalar{255, 255, 0});
  //   }
  // }
  // cv::imwrite("test_vision_infer_out.jpg", image_bgr);
  
  
  cv::imwrite("test_vision_infer_nv12.jpg", image_nv12);
  for (auto &c : *chars) {
    std::cout << c << ", ";
  }
  std::cout << std::endl;
  
  gflags::ShutDownCommandLineFlags();
  return 0;
}
