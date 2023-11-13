/**
 * @file faceRecognition.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 人脸识别逻辑串联，输入单帧图像，输出最中心人脸特征
 * @version 0.1
 * @date 2023-10-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "preprocess.hpp"
#include "visionInfer.hpp"
#include <atomic>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#ifndef __SERVER_FACE_CORE_FACE_RECOGNITION_HPP_
#define __SERVER_FACE_CORE_FACE_RECOGNITION_HPP_
namespace server::face::core {

using namespace infer;
using namespace common;

using algo_ptr = std::shared_ptr<AlgoInfer>;

class FaceRecognition {
public:
  FaceRecognition() {
    std::string detModelPath = "/opt/deploy/models/yolov5n-face-sim.engine";
    std::string recModelPath = "/opt/deploy/models/arcface_112x112_nv12.engine";
    PointsDetAlgo faceDet_config{{
                                     1,
                                     {"input"},
                                     {"output"},
                                     detModelPath,
                                     "YoloPDet",
                                     {640, 640, 3},
                                     false,
                                     255.0,
                                     0,
                                     0.3,
                                 },
                                 5,
                                 0.4};
    AlgoConfig fdet_config;
    fdet_config.setParams(faceDet_config);

    FeatureAlgo faceNet_config{{
                                   1,
                                   {"input.1"},
                                   {"516"},
                                   recModelPath,
                                   "FaceNet",
                                   {112, 112, 3},
                                   false,
                                   127.5,
                                   1.0,
                                   0.3,
                               },
                               512};
    AlgoConfig frec_config;
    frec_config.setParams(faceNet_config);

    faceDet = getVision(std::move(fdet_config));
    faceRec = getVision(std::move(frec_config));
  }
  ~FaceRecognition() {}

  bool forward(FrameInfo &inputData, std::vector<float> &feature) {

    // 人脸检测
    InferResult faceDetRet;
    inference(inputData, faceDetRet, faceDet);

    auto kbboxes = std::get_if<KeypointsBoxes>(&faceDetRet.aRet);
    if (!kbboxes || kbboxes->empty()) {
      FLOWENGINE_LOGGER_INFO("Not a single face was detected!");
      return false;
    }

    // 获取最靠近中心的人脸
    size_t index = findClosestBBoxIndex(*kbboxes, inputData.shape.at(0),
                                        inputData.shape.at(1));

    auto kbbox = kbboxes->at(index);

    // 构造输入图像
    cv::Mat image{inputData.inputShape.at(1), inputData.inputShape.at(0),
                  inputData.inputShape.at(2) == 1 ? CV_8UC1 : CV_8UC3,
                  reinterpret_cast<char *>(*inputData.data)};

    // 获取人脸识别输入图像
    FrameInfo faceInput;
    cv::Mat faceImage;
    getFaceInput(image, faceImage, faceInput, kbbox.points, inputData.type);
    // cv::imwrite("output.jpg", faceImage);
    // 人脸特征提取
    InferResult faceRecRet;
    inference(faceInput, faceRecRet, faceRec);
    feature = *std::get_if<std::vector<float>>(&faceRecRet.aRet);
    utils::normalize_L2(feature.data(), feature.size());
    return true;
  }

private:
  std::atomic_bool status = false;
  algo_ptr faceDet;
  algo_ptr faceRec;

private:
  algo_ptr getVision(AlgoConfig &&config) {

    std::shared_ptr<AlgoInfer> vision = std::make_shared<VisionInfer>(config);
    if (!vision->init()) {
      FLOWENGINE_LOGGER_ERROR("Failed to init vision");
      std::exit(-1); // 强制中断
      return nullptr;
    }
    return vision;
  }

  void inference(FrameInfo &frame, InferResult &ret, algo_ptr vision) {
    InferParams params{std::string("xxx"),
                       frame.type,
                       0.0,
                       {"xxx"},
                       {frame.inputShape.at(0), frame.inputShape.at(1),
                        frame.inputShape.at(2)}};

    vision->infer(frame, params, ret);
  }

  cv::Mat estimateNorm(const std::vector<cv::Point2f> &landmarks,
                       int imageSize = 112) {
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

  cv::Mat normCrop(const cv::Mat &img,
                   const std::vector<cv::Point2f> &landmarks,
                   int imageSize = 112) {
    cv::Mat M = estimateNorm(landmarks, imageSize);
    cv::Mat warped;
    cv::warpAffine(img, warped, M, cv::Size(imageSize, imageSize),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
    return warped;
  }

  void getFaceInput(cv::Mat const &input, cv::Mat &output, FrameInfo &frame,
                    Points2f const &points, ColorType const &type) {
    // 关键点矫正
    std::vector<cv::Point2f> cvPoints;
    for (auto &p : points) {
      cvPoints.push_back(cv::Point2f{p.x, p.y});
    }
    if (type == ColorType::NV12) {
      cv::Mat temp;
      utils::NV12toRGB(input, temp);
      temp = normCrop(temp, cvPoints, 112);
      utils::RGB2NV12(temp, output);
      frame.shape = {temp.cols, temp.rows, output.channels()};
    } else {
      output = normCrop(input, cvPoints, 112);
      frame.shape = {output.cols, output.rows, output.channels()};
    }
    frame.inputShape = {output.cols, output.rows, output.channels()};
    frame.type = type;
    frame.data = reinterpret_cast<void **>(&output.data);
  }

  size_t findClosestBBoxIndex(KeypointsBoxes const &kbboxes, float w, float h) {
    float image_center_x = w / 2.0;
    float image_center_y = h / 2.0;

    float min_distance = std::numeric_limits<float>::max();
    size_t closest_bbox_index = -1;

    for (size_t i = 0; i < kbboxes.size(); ++i) {
      const auto &kbbox = kbboxes[i];
      float bbox_center_x = (kbbox.bbox.bbox[0] + kbbox.bbox.bbox[2]) / 2.0;
      float bbox_center_y = (kbbox.bbox.bbox[1] + kbbox.bbox.bbox[3]) / 2.0;

      float distance = std::hypot(bbox_center_x - image_center_x,
                                  bbox_center_y - image_center_y);

      if (distance < min_distance) {
        min_distance = distance;
        closest_bbox_index = i;
      }
    }
    return closest_bbox_index;
  }
};
} // namespace server::face::core
#endif
