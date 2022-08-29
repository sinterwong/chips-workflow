#include "inference.h"
#include "jetson/assdDet.hpp"
#include "jetson/detection.hpp"
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <gflags/gflags.h>

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

  cv::Mat image = cv::imread(FLAGS_image_path);

  std::vector<std::string> inputNames = {"data"};
  std::vector<std::string> outputNames = {"output2_conv3", "output3_conv3",
                                          "output4_conv3"};
  std::array<int, 3> inputShape{FLAGS_input_width, FLAGS_input_height, 3};
  common::AlgorithmConfig params{FLAGS_model_path,
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
  result.shape = {95, 191};
  det->infer(image.data, result);
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
./test_assd --image_path=/home/wangxt/workspace/projects/flowengine/tests/data/person.jpg \
            --model_path=/home/wangxt/workspace/projects/flowengine/tests/data/M-01-01-0001-oil_head_head.onnx \
            --input_height=191 \
            --input_width=95
*/
