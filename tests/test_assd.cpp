#include "inference.h"
#include "jetson/assdDet.hpp"
#include "jetson/detection.hpp"
#include "jetson/yoloDet.hpp"
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

void hwc_to_chw(cv::InputArray &src, cv::OutputArray &dst) {
  const int src_h = src.rows();
  const int src_w = src.cols();
  const int src_c = src.channels();

  cv::Mat hw_c = src.getMat().reshape(1, src_h * src_w);

  const std::array<int, 3> dims = {src_c, src_h, src_w};
  dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));
  cv::Mat dst_1d = dst.getMat().reshape(1, {src_c, src_h, src_w});

  cv::transpose(hw_c, dst_1d);
}

void chw_to_hwc(cv::InputArray &src, cv::OutputArray &dst) {
  const auto &src_size = src.getMat().size;
  const int src_c = src_size[0];
  const int src_h = src_size[1];
  const int src_w = src_size[2];

  auto c_hw = src.getMat().reshape(0, {src_c, src_h * src_w});

  dst.create(src_h, src_w, CV_MAKETYPE(src.depth(), src_c));
  cv::Mat dst_1d = dst.getMat().reshape(src_c, {src_h, src_w});

  cv::transpose(c_hw, dst_1d);
}

int main(int argc, char **argv) {

  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> inputNames = {"data"};
  // std::vector<std::string> inputNames = {"images"};
  // std::vector<std::string> outputNames = {"output2_conv3", "output3_conv3",
  //                                         "output4_conv3"};
  std::vector<std::string> outputNames = {"conv10_3", "conv13_3", "conv15_3", "conv16_3"};
  // std::vector<std::string> outputNames = {"output"};

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
  // std::shared_ptr<DetectionInfer> det = std::make_shared<YoloDet>(params);
  det->initialize();

  cv::Mat image = cv::imread(FLAGS_image_path);
  // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  // cv::Mat chw_image;
  // hwc_to_chw(image, chw_image);

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
