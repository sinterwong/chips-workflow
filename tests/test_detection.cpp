#include "detection.hpp"
#include "infer_utils.hpp"
#include "inference.h"
#include <gflags/gflags.h>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#if (TARGET_PLATFORM == 0)
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#elif (TARGET_PLATFORM == 1)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
#endif
// using namespace infer::vision;

DEFINE_string(image_path, "", "Specify config path.");
DEFINE_string(model_path, "", "Specify model path.");
DEFINE_string(atype, "assd", "Specify algorithim type.");
DEFINE_int32(input_height, 335, "Specify input height.");
DEFINE_int32(input_width, 335, "Specify input width.");

int main(int argc, char **argv) {

  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // TODO 根据不同平台，给出不同的类型
  cv::Mat image = cv::imread(FLAGS_image_path);

  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  float alpha = 0;
  if (FLAGS_atype == "Yolo") {
    inputNames = {"images"};
    outputNames = {"output"};
    alpha = 255.0;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  } else if (FLAGS_atype == "Assd") {
    inputNames = {"data"};
    outputNames = {"conv13_3", "conv15_3", "conv16_3"};
  } else {
    return -1;
  }

  std::array<int, 3> inputShape{FLAGS_input_width, FLAGS_input_height, 3};
  common::AlgorithmConfig params{FLAGS_model_path,
                                 std::move(inputNames),
                                 std::move(outputNames),
                                 std::move(inputShape),
                                 FLAGS_atype, // yolo, assd
                                 0.3,
                                 0.4,
                                 alpha,
                                 0,
                                 false,
                                 1};
  std::shared_ptr<AlgoInference> instance =
      std::make_shared<AlgoInference>(params);
  if (!instance->initialize()) {
    FLOWENGINE_LOGGER_ERROR("initialization is failed!");
    return -1;
  }
  std::cout << "instance initialize has done!" << std::endl;
  infer::ModelInfo info;
  instance->getModelInfo(info);
  std::shared_ptr<infer::vision::Detection> det;
  det = ObjectFactory::createObject<infer::vision::Detection>(FLAGS_atype,
                                                              params, info);
  if (det == nullptr) {
    FLOWENGINE_LOGGER_ERROR("Error algorithm serial {}", FLAGS_atype);
    return -1;
  }
  infer::Result ret;
  ret.shape = {image.cols, image.rows, 3};
  infer::FrameInfo frame;
  cv::Mat data;
  char *d[3] = {0, 0, 0};
  if (TARGET_PLATFORM == 0) {
    infer::utils::RGB2NV12(image, data);
    d[0] = reinterpret_cast<char *>(data.data);
    d[1] = reinterpret_cast<char *>(data.data) + (image.cols * image.rows);
    frame.data = reinterpret_cast<void **>(d);
    cv::Mat picNV12 = cv::Mat(image.rows * 3 / 2, image.cols, CV_8UC1, d[0]);
    cv::Mat temp_y = cv::Mat(image.rows, image.cols, CV_8UC1, d[0]);
    cv::Mat temp_uv = cv::Mat(image.rows / 2, image.cols, CV_8UC1, d[1]);
    cv::imwrite("nv12_finial.png", picNV12);
    cv::imwrite("y_finial.png", temp_y);
    cv::imwrite("uv_finial.png", temp_uv);
  } else {
    frame.data = reinterpret_cast<void **>(&image.data);
    data = image.clone();
  }
  cv::imwrite("test_detection_data.jpg", data);

  frame.shape = ret.shape;
  std::cout << "image shape: " << image.cols << ", " << image.rows << std::endl;

  void *outputs[info.output_count];
  void *output = reinterpret_cast<void *>(outputs);
  instance->infer(frame, &output);
  std::cout << "infer has done!" << std::endl;
  float **out = reinterpret_cast<float **>(output);
  for (int i = 5000; i < 5050; i++) {
    std::cout << out[0][i] << ", ";
  }
  std::cout << std::endl;

  det->processOutput(&output, ret);
  FLOWENGINE_LOGGER_INFO("number of result: {}", ret.detResults.size());
  for (auto &bbox : ret.detResults) {
    cv::Rect rect(bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
                  bbox.bbox[3] - bbox.bbox[1]);
    cv::rectangle(image, rect, cv::Scalar(0, 0, 255), 2);
  }
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
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
