#include "inference.h"
#include "jetson/assdDet.hpp"
#include "jetson/detection.hpp"
#include <memory>
#include <opencv2/imgcodecs.hpp>

using namespace infer::trt;

int main() {
  std::string imPath = "/home/wangxt/workspace/projects/flowengine/tests/data/person2.png"; 
  cv::Mat image = cv::imread(imPath);

  std::string modelPath = "/home/wangxt/workspace/projects/flowengine/tests/"
                          "data/assd_shoulder.engine";
  std::vector<std::string> inputNames = {"data"};
  std::vector<std::string> outputNames = {"output2_conv3", "output3_conv3",
                                          "output4_conv3"};
  std::array<int, 3> inputShape{335, 335, 3};
  common::AlgorithmConfig params{modelPath,
                                 std::move(inputNames),
                                 std::move(outputNames),
                                 std::move(inputShape),
                                 1,
                                 0,
                                 0.4,
                                 0.4,
                                 0,
                                 0,
                                 false,
                                 1};
  std::shared_ptr<DetectionInfer> det = std::make_shared<AssdDet>(params);
  det->initialize();

  infer::Result result;
  det->infer(image.data, result);
  for (auto &bbox : result.detResults) {
    cv::Rect rect(bbox.bbox[0], bbox.bbox[1],
                  bbox.bbox[2] - bbox.bbox[0],
                  bbox.bbox[3] - bbox.bbox[1]);
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
  }
  cv::imwrite("test_assd_out.jpg", image);
  return 0;
}