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

  bool forward(cv::Mat &image, std::vector<float> &feature) {
    // // TODO 模拟特征生成
    // for (int i = 0; i < ndim; ++i) {
    //   feature.push_back(rand() % 1000);
    // }
    // infer::utils::normalize_L2(feature.data(), ndim);

    auto ctype = getColorType(image);
    int imageWidth, imageHeight;
    imageWidth = image.cols;
    if (ctype == ColorType::NV12) {
      imageHeight = image.rows * 2 / 3;
    } else {
      imageHeight = image.rows;
    }

    InferResult faceDetRet;
    inference(image, faceDetRet, faceDet);
    auto kbboxes = std::get_if<KeypointsBoxes>(&faceDetRet.aRet);
    if (!kbboxes || kbboxes->empty()) {
      FLOWENGINE_LOGGER_INFO("Not a single face was detected!");
      return -1;
    }

    size_t index = findClosestBBoxIndex(*kbboxes, imageWidth, imageHeight);

    auto kbbox = kbboxes->at(index);

    cv::Mat faceInput;
    cv::Rect2i rect{static_cast<int>(kbbox.bbox.bbox[0]),
                    static_cast<int>(kbbox.bbox.bbox[1]),
                    static_cast<int>(kbbox.bbox.bbox[2] - kbbox.bbox.bbox[0]),
                    static_cast<int>(kbbox.bbox.bbox[3] - kbbox.bbox.bbox[1])};
    // 可视化人脸检测结果：人脸框和5个关键点
    infer::utils::cropImage(image, faceInput, rect, ctype);
    std::vector<cv::Point2f> points;
    for (auto &p : kbbox.points) {
      points.push_back(cv::Point2f{p.x, p.y});
    }
    // 关键点矫正
    if (ctype == ColorType::NV12) {
      cv::Mat temp;
      utils::NV12toRGB(image, temp);
      faceInput = normCrop(temp, points, 112);
      utils::RGB2NV12(temp, faceInput);
    } else {
      faceInput = normCrop(image, points, 112);
    }

    // 人脸特征提取
    InferResult faceRecRet;
    inference(faceInput, faceRecRet, faceRec);
    feature = std::move(std::get<std::vector<float>>(faceRecRet.aRet));
    utils::normalize_L2(feature.data(), 512);
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

  void inference(cv::Mat &image, InferResult &ret, algo_ptr vision) {
    ColorType ctype = getColorType(image);
    if (ctype != ColorType::NV12) {
      // 默认是RGB（垃圾代码，不要在意）
      utils::RGB2NV12(image, image);
    }
    RetBox region{"xxx"};

    InferParams params{std::string("xxx"),
                       ColorType::NV12,
                       0.0,
                       region,
                       {image.cols, image.rows, image.channels()}};

    // 制作输入数据
    FrameInfo frame;
    switch (params.frameType) {
    case common::ColorType::None:
    case common::ColorType::RGB888:
    case common::ColorType::BGR888:
      frame.shape = {image.cols, image.rows, 3};
      break;
    case common::ColorType::NV12:
      frame.shape = {image.cols, image.rows * 2 / 3, 3};
      break;
    }
    frame.shape = {image.cols, image.rows * 2 / 3, 3};
    frame.type = params.frameType;
    frame.data = reinterpret_cast<void **>(&image.data);
    vision->infer(frame, params, ret);
  }

  ColorType getColorType(cv::Mat &image) {
    if (image.channels() == 1) {
      return ColorType::NV12;
    } else {
      return ColorType::RGB888;
    }
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
