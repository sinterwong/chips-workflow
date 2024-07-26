/**
 * @file face_recognition.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-31
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "logger/logger.hpp"
#include "preprocess.hpp"
#include "visionInfer.hpp"
#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <vector>

DEFINE_string(img, "", "Specify a image which contains some face path.");
DEFINE_string(det_model_path, "", "Specify the model path of FaceDet.");
DEFINE_string(rec_model_path, "", "Specify the model path of FaceRec.");

using namespace common;
using algo_ptr = std::shared_ptr<AlgoInfer>;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

algo_ptr getVision(AlgoConfig &&config) {

  std::shared_ptr<AlgoInfer> vision =
      std::make_shared<infer::VisionInfer>(config);
  if (!vision->init()) {
    FLOWENGINE_LOGGER_ERROR("Failed to init vision");
    std::exit(-1); // 强制中断
    return nullptr;
  }
  return vision;
}

void inference(cv::Mat &image, InferResult &ret,
               std::shared_ptr<AlgoInfer> vision) {

  RetBox region{"hello"};

  InferParams params{std::string("hello"),
                     ColorType::NV12,
                     0.0,
                     region,
                     {image.cols, image.rows, image.channels()}};

  // 制作输入数据
  FrameInfo frame;
  frame.shape = {image.cols, image.rows * 2 / 3, 3};
  frame.type = params.frameType;
  frame.data = reinterpret_cast<void **>(&image.data);
  vision->infer(frame, params, ret);
}

cv::Mat estimateNorm(const std::vector<cv::Point2f> &landmarks,
                     int imageSize = 112, const std::string &mode = "arcface") {
  assert(landmarks.size() == 5);
  assert(imageSize % 112 == 0 || imageSize % 128 == 0);

  float ratio;
  float diffX = 0.0;
  if (imageSize % 112 == 0) {
    ratio = static_cast<float>(imageSize) / 112.0f;
  } else {
    ratio = static_cast<float>(imageSize) / 128.0f;
    diffX = 8.0f * ratio;
  }

  // Assuming arcfaceDst is a predefined 5x2 matrix of facial landmarks for
  // normalization You need to define this matrix based on your specific use
  // case
  cv::Mat arcfaceDst =
      (cv::Mat_<float>(5, 2) << 38.2946f, 51.6963f, 73.5318f, 51.5014f,
       56.0252f, 71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f);

  cv::Mat dst = arcfaceDst * ratio;
  for (int i = 0; i < dst.rows; ++i) {
    dst.at<float>(i, 0) += diffX;
  }

  cv::Mat src(landmarks);
  cv::Mat tform = cv::estimateAffinePartial2D(src, dst);
  return tform;
}

cv::Mat normCrop(const cv::Mat &img, const std::vector<cv::Point2f> &landmarks,
                 int imageSize = 112, const std::string &mode = "arcface") {
  cv::Mat M = estimateNorm(landmarks, imageSize, mode);
  cv::Mat warped;
  cv::warpAffine(img, warped, M, cv::Size(imageSize, imageSize),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
  return warped;
}

int main(int argc, char **argv) {
  gflags::SetUsageMessage("Face recognition");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FlowEngineLoggerSetLevel(1);

  PointsDetAlgo faceDet_config{{
                                   "faceDet",
                                   1,
                                   {"images"},
                                   {"output"},
                                   FLAGS_det_model_path,
                                   "YoloPDet",
                                   {640, 640, 3},
                                   false,
                                   0,
                                   0,
                                   0.3,
                               },
                               5,
                               0.4};
  AlgoConfig fdet_config;
  fdet_config.setParams(faceDet_config);

  FeatureAlgo faceNet_config{{
                                 "faceRec",
                                 1,
                                 {"images"},
                                 {"output"},
                                 FLAGS_rec_model_path,
                                 "FaceNet",
                                 {112, 112, 3},
                                 false,
                                 255.0,
                                 0,
                                 0.3,
                             },
                             512};
  AlgoConfig frec_config;
  frec_config.setParams(faceNet_config);

  auto faceDet = getVision(std::move(fdet_config));
  auto faceRec = getVision(std::move(frec_config));

  // 图片读取
  cv::Mat image_bgr = cv::imread(FLAGS_img);

  cv::Mat image_rgb;
  cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

  cv::Mat image_nv12;
  infer::utils::RGB2NV12(image_rgb, image_nv12);

  InferResult faceDetRet;
  inference(image_nv12, faceDetRet, faceDet);
  auto kbboxes = std::get_if<KeypointsBoxes>(&faceDetRet.aRet);
  if (!kbboxes) {
    FLOWENGINE_LOGGER_ERROR("Not a single face was detected!");
    return -1;
  }

  std::vector<Eigenvector> features;

  for (auto &kbbox : *kbboxes) {
    cv::Mat ori_face_nv12, ori_face_rgb;
    cv::Rect2i rect{static_cast<int>(kbbox.bbox.bbox[0]),
                    static_cast<int>(kbbox.bbox.bbox[1]),
                    static_cast<int>(kbbox.bbox.bbox[2] - kbbox.bbox.bbox[0]),
                    static_cast<int>(kbbox.bbox.bbox[3] - kbbox.bbox.bbox[1])};
    // 可视化人脸检测结果：人脸框和5个关键点
    cv::rectangle(image_bgr, rect, cv::Scalar(0, 0, 255), 2);
    infer::utils::cropImage(image_nv12, ori_face_nv12, rect, ColorType::NV12);
    int i = 1;
    std::vector<cv::Point2f> points;
    for (auto &p : kbbox.points) {
      cv::circle(image_bgr,
                 cv::Point{static_cast<int>(p.x), static_cast<int>(p.y)}, 3,
                 cv::Scalar{255, 255, 0});
      cv::putText(image_bgr, std::to_string(i++),
                  cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 14, 50), 1);
      points.push_back(cv::Point2f{p.x, p.y});
    }

    // TODO 基于5个点的人脸关键点矫正
    std::string prefix = std::to_string(rand() % 1000);
    infer::utils::NV12toRGB(ori_face_nv12, ori_face_rgb);
    cv::imwrite(prefix + "_ori_face.jpg", ori_face_rgb);

    cv::Mat aligned_face_rgb = normCrop(image_rgb, points, 112);
    cv::imwrite(prefix + "_aligned_face.jpg", aligned_face_rgb);

    cv::Mat aligned_face_nv12;
    infer::utils::RGB2NV12(aligned_face_rgb, aligned_face_nv12);

    // TODO 人脸特征提取
    InferResult faceRecRet;
    inference(aligned_face_nv12, faceRecRet, faceRec);
    auto feature = std::get_if<Eigenvector>(&faceRecRet.aRet);
    features.push_back(*feature);
  }

  if (features.size() >= 2) {
    // 彼此可以计算相似度
    Eigen::MatrixXf v1 =
        Eigen::Map<Eigen::Matrix<float, 1, 512>>(features.at(0).data());
    Eigen::MatrixXf v2 =
        Eigen::Map<Eigen::Matrix<float, 1, 512>>(features.at(1).data());

    // 计算v1和v2之间的cosine相似度
    Eigen::MatrixXf norm_v1 = v1.rowwise().norm();
    Eigen::MatrixXf norm_v2 = v2.rowwise().norm();

    // 将分母中为0的元素置为一个很小的非零值
    norm_v1 = (norm_v1.array() == 0).select(1e-8, norm_v1);
    norm_v2 = (norm_v2.array() == 0).select(1e-8, norm_v2);

    Eigen::MatrixXf dot_product = v1 * v2.transpose(); // 1xn

    Eigen::MatrixXf cosine_sim =
        dot_product.array() / (norm_v1 * norm_v2.transpose()).array();

    // 输出结果
    std::cout << "The cosine similar is: " << cosine_sim << std::endl;
  }

  cv::imwrite("test_face_rec_out.jpg", image_bgr);

  gflags::ShutDownCommandLineFlags();
  return 0;
}
