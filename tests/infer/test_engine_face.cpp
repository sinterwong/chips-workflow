#include <gflags/gflags.h>
#include <iostream>

#include <Eigen/Dense>

#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include "preprocess.hpp"
#include "visionInfer.hpp"

DEFINE_string(img1, "", "Specify face1 image path.");
DEFINE_string(img2, "", "Specify face2 image path.");
DEFINE_string(model_path, "", "Specify the yolo model path.");

using namespace infer;
using namespace common;

std::vector<float> getFeature(std::string &imPath,
                              std::shared_ptr<AlgoInfer> vision) {
  cv::Mat image_bgr = cv::imread(imPath);
  cv::Mat image_rgb, image_nv12;
  cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
  infer::utils::RGB2NV12(image_rgb, image_nv12);
  RetBox region{"hello"};

  InferParams params{std::string("hello"),
                     ColorType::NV12,
                     0.0,
                     region,
                     {image_nv12.cols, image_nv12.rows, image_nv12.channels()}};
  InferResult ret;

  FrameInfo frame;
  frame.shape = {image_nv12.cols, image_nv12.rows * 2 / 3, 3};
  frame.type = params.frameType;
  frame.data = reinterpret_cast<void **>(&image_nv12.data);
  vision->infer(frame, params, ret);

  auto feature = std::get_if<Eigenvector>(&ret.aRet);
  if (!feature) {
    FLOWENGINE_LOGGER_ERROR("Wrong algorithm type!");
  }
  return *feature;
}

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  float alpha = 0;
  inputNames = {"images"};
  outputNames = {"output"};
  alpha = 255.0;
  common::Shape inputShape{160, 160, 3};
  common::AlgoBase base_config{
      "facenet",
      1,
      std::move(inputNames),
      std::move(outputNames),
      FLAGS_model_path,
      "FaceNet",
      std::move(inputShape),
      false,
      alpha,
      0,
      0.3,
  };

  // DetAlgo det_config{std::move(base_config), 0.4};
  common::FeatureAlgo config{std::move(base_config), 128};

  AlgoConfig center;

  // center.setParams(det_config);
  center.setParams(config);

  std::shared_ptr<AlgoInfer> vision = std::make_shared<VisionInfer>(center);
  if (!vision->init()) {
    FLOWENGINE_LOGGER_ERROR("Failed to init vision");
    return -1;
  }

  auto feature1 = getFeature(FLAGS_img1, vision);
  auto feature2 = getFeature(FLAGS_img2, vision);

  Eigen::MatrixXf v1 =
      Eigen::Map<Eigen::Matrix<float, 1, 128>>(feature1.data());
  Eigen::MatrixXf v2 =
      Eigen::Map<Eigen::Matrix<float, 1, 128>>(feature2.data());

  // Eigen::MatrixXf v1(1, 16);
  // Eigen::MatrixXf v2(10, 16);

  // v1.setRandom();
  // v2.setRandom();

  // 计算v1和v2之间的cosine相似度
  Eigen::MatrixXf norm_v1 = v1.rowwise().norm();
  Eigen::MatrixXf norm_v2 = v2.rowwise().norm();

  // 将分母中为0的元素置为一个很小的非零值
  norm_v1 = (norm_v1.array() == 0).select(1e-8, norm_v1);
  norm_v2 = (norm_v2.array() == 0).select(1e-8, norm_v2);

  Eigen::MatrixXf dot_product = v1 * v2.transpose(); // 1x10

  Eigen::MatrixXf cosine_sim =
      dot_product.array() / (norm_v1 * norm_v2.transpose()).array();

  // 输出结果
  std::cout << v1 << std::endl;
  std::cout << v2 << std::endl;
  std::cout << cosine_sim << std::endl;

  gflags::ShutDownCommandLineFlags();

  return 0;
}

/*
./test_face --img1 /root/workspace/softwares/flowengine/data/face1.jpg \
            --img2 /root/workspace/softwares/flowengine/data/face2.jpg \
            --model_path
/root/workspace/softwares/flowengine/models/facenet_mobilenet_160x160.bin
*/