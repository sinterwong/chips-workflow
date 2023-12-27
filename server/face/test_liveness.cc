/**
 * @file test_liveness.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-28
 *
 * @copyright Copyright (c) 2023
 *
 */

/**
 * @file test_face_quality.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "algoManager.hpp"
#include "core/algoDispatcher.hpp"
#include "logger/logger.hpp"
#include "postprocess.hpp"
#include "videoDecode.hpp"
#include <cstddef>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <vector>

DEFINE_string(video, "", "Specify a video url.");

using namespace server::face::core;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

constexpr int KEY_POINTS_INPUT_SIZE = 192;

cv::Mat transform(int size, cv::Point2f center, double scale, double rotation) {
  // Step 1: 缩放变换
  cv::Mat scaleMat = cv::getRotationMatrix2D(center, 0, scale);
  cv::Mat scaleMat3x3 = cv::Mat::eye(3, 3, scaleMat.type());
  scaleMat.copyTo(scaleMat3x3.rowRange(0, 2));

  // Step 2: 向原始中心的负方向平移
  cv::Mat transToNegCenterMat = cv::Mat::eye(3, 3, scaleMat.type());
  transToNegCenterMat.at<double>(0, 2) = -center.x;
  transToNegCenterMat.at<double>(1, 2) = -center.y;

  // Step 3: 旋转变换
  cv::Mat rotMat = cv::getRotationMatrix2D(cv::Point2f(0, 0), rotation, 1);
  cv::Mat rotMat3x3 = cv::Mat::eye(3, 3, rotMat.type());
  rotMat.copyTo(rotMat3x3.rowRange(0, 2));

  // Step 4: 向图像中心的平移
  cv::Mat transToImgCenterMat = cv::Mat::eye(3, 3, rotMat.type());
  transToImgCenterMat.at<double>(0, 2) = size / 2.0;
  transToImgCenterMat.at<double>(1, 2) = size / 2.0;

  // 组合所有变换
  cv::Mat combinedMat3x3 =
      transToImgCenterMat * rotMat3x3 * transToNegCenterMat * scaleMat3x3;

  // 将3x3矩阵转换回2x3矩阵
  cv::Mat combinedMat = combinedMat3x3.rowRange(0, 2);

  combinedMat.convertTo(combinedMat, CV_32FC1);

  return combinedMat;
}

cv::Mat normCrop(cv::Mat const &image, cv::Mat const &rotMat, int size) {
  cv::Mat aligned_face;
  cv::warpAffine(image, aligned_face, rotMat, cv::Size(size, size));

  return aligned_face;
}

float get2PointsNorm(Point2f const &p1, Point2f const &p2) {
  return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
}

float getEyeAspectRatio(std::vector<Point2f> &points) {
  float directionH1 = get2PointsNorm(points.at(0), points.at(7));
  float directionH2 = get2PointsNorm(points.at(3), points.at(8));
  float directionH3 = get2PointsNorm(points.at(4), points.at(9));
  float directionV1 = get2PointsNorm(points.at(2), points.at(6));

  return (directionH1 + directionH2 + directionH3) / (3 * directionV1);
}

float eyeAspectRatio(std::vector<Point2f> &points) {
  // 左眼对应的点
  std::vector<Point2f> leftEyePoints;
  for (int i = 33; i < 43; ++i) {
    leftEyePoints.push_back(points[i]);
  }

  // 右眼对应的点
  std::vector<Point2f> rightEyePoints;
  for (int i = 87; i < 97; ++i) {
    rightEyePoints.push_back(points[i]);
  }

  // 计算左眼的纵横比
  float leftEyeAspectRatio = getEyeAspectRatio(leftEyePoints);

  // 计算右眼的纵横比
  float rightEyeAspectRatio = getEyeAspectRatio(rightEyePoints);

  return (leftEyeAspectRatio + rightEyeAspectRatio) / 2;
}

float getMouthAspectRatio(std::vector<Point2f> &points) {
  float directionH1 = get2PointsNorm(points.at(1), points.at(13));
  float directionH2 = get2PointsNorm(points.at(7), points.at(9));
  float directionH3 = get2PointsNorm(points.at(4), points.at(17));
  float directionV1 = get2PointsNorm(points.at(12), points.at(16));
  return (directionH1 + directionH2 + directionH3) / directionV1;
}

float mouthAspectRatio(std::vector<Point2f> &points) {
  // 嘴巴对应的点
  std::vector<Point2f> mouthPoints;
  for (int i = 53; i < 72; ++i) {
    mouthPoints.push_back(points[i]);
  }

  // 计算嘴巴的纵横比
  return getMouthAspectRatio(mouthPoints);
}

enum class LivenessStatus { None = 0, MOUTH, EYE, NOD, SHAKE };

int main(int argc, char **argv) {
  FlowEngineLoggerSetLevel(1);
  gflags::SetUsageMessage("Face recognition");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 算法启动
  AlgoManager &algoManager = AlgoManager::getInstance();

  // 视频启动
  std::unique_ptr<video::VideoDecode> decoder =
      std::make_unique<video::VideoDecode>();
  if (!decoder->init()) {
    FLOWENGINE_LOGGER_ERROR("init decoder failed!");
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");

  if (!decoder->start(FLAGS_video)) {
    FLOWENGINE_LOGGER_ERROR("run decoder failed!");
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("Video manager is running!");

  // 人脸初始状态是否就绪
  bool isFaceReady = false;
  // 成功判定的次数
  int successCount = 0;

  // 动作执行状态
  bool actionStatus = false; // false表示未执行，true表示执行中

  // 当前执行动作的类型
  LivenessStatus actionType = LivenessStatus::None;

  std::vector<LivenessStatus> actionTypes{
      LivenessStatus::MOUTH, LivenessStatus::EYE, LivenessStatus::NOD,
      LivenessStatus::SHAKE};
  int count = 0;
  while (decoder->isRunning() && ++count < 1000 && successCount < 2) {
    auto image = decoder->getcvImage();
    if (count % 5 != 0) {
      continue;
    }
    if (!image) {
      FLOWENGINE_LOGGER_ERROR("get image failed!");
      continue;
    }

    if (!actionStatus) { // 如果当前没有执行动作，则随机选取一个动作
      // 随机选取一个动作，选完后剔除
      int index = rand() % actionTypes.size();
      actionType = actionTypes[index];
      actionTypes.erase(actionTypes.begin() + index);
      actionStatus = true;
    }

    // 获取bgr
    cv::Mat image_bgr;
    cv::cvtColor(*image, image_bgr, cv::COLOR_YUV2BGR_NV12);

    // 人脸检测
    FrameInfo frame;
    frame.data = reinterpret_cast<void **>(&image->data);
    frame.inputShape = {image->cols, image->rows, image->channels()};
    frame.shape = {image->cols, image->rows * 2 / 3, image->channels()};
    frame.type = ColorType::NV12;
    InferResult detRet;
    auto faceDet = algoManager.infer(AlgoType::DET, frame, detRet);
    if (!faceDet.get()) {
      FLOWENGINE_LOGGER_INFO("Face detection failed!");
      continue;
    }

    auto kbboxes = std::get_if<KeypointsBoxes>(&detRet.aRet);
    if (!kbboxes || kbboxes->empty()) {
      FLOWENGINE_LOGGER_INFO("Not a single face was detected!");
      continue;
    }

    // 获取最靠近中心的人脸
    size_t aIndex =
        utils::findClosestBBoxIndex(*kbboxes, image->cols, image->rows);
    auto kbbox = kbboxes->at(aIndex);

    {
      /* 可视化人脸检测结果：人脸框和5个关键点 */
      cv::Rect2i rect{
          static_cast<int>(kbbox.bbox.bbox[0]),
          static_cast<int>(kbbox.bbox.bbox[1]),
          static_cast<int>(kbbox.bbox.bbox[2] - kbbox.bbox.bbox[0]),
          static_cast<int>(kbbox.bbox.bbox[3] - kbbox.bbox.bbox[1])};
      cv::rectangle(image_bgr, rect, cv::Scalar(0, 0, 255), 2);
      int i = 1;
      std::vector<cv::Point2f> points;
      for (auto &p : kbbox.points) {
        cv::circle(image_bgr,
                   cv::Point{static_cast<int>(p.x), static_cast<int>(p.y)}, 4,
                   cv::Scalar{0, 0, 255});
        cv::putText(image_bgr, std::to_string(i++),
                    cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 14, 50), 1);
        points.push_back(cv::Point2f{p.x, p.y});
      }
    }

    // 根据人脸质检结果获取初始状态
    if (!isFaceReady) {
      // 人脸质量检测
      FrameInfo faceInput;
      cv::Mat faceImage;
      int x, y, w, h;
      x = static_cast<int>(kbbox.bbox.bbox[0]);
      y = static_cast<int>(kbbox.bbox.bbox[1]);
      w = static_cast<int>(kbbox.bbox.bbox[2]) - x;
      h = static_cast<int>(kbbox.bbox.bbox[3]) - y;

      cv::Rect retBox{x, y, w, h};
      if (!utils::cropImage(*image, faceImage, retBox, ColorType::NV12)) {
        FLOWENGINE_LOGGER_ERROR("Crop image failed!");
        return -1;
      }

      faceInput.data = reinterpret_cast<void **>(&faceImage.data);
      faceInput.inputShape = {faceImage.cols, faceImage.rows,
                              faceImage.channels()};
      faceInput.shape = {faceImage.cols, faceImage.rows * 2 / 3,
                         faceImage.channels()};
      faceInput.type = ColorType::NV12;
      InferResult qualityRet;
      auto faceQuality =
          algoManager.infer(AlgoType::QUALITY, faceInput, qualityRet);
      if (!faceQuality.get()) {
        FLOWENGINE_LOGGER_INFO("Face quality detection failed!");
        continue;
      }
      auto quality = std::get_if<ClsRet>(&qualityRet.aRet);
      if (!quality) {
        FLOWENGINE_LOGGER_INFO("Face quality detection failed!");
        continue;
      }
      // TODO:此处为了方便实现活体功能仅使用了遮挡算法的结果，正式使用时需要结合图片质量和检测质量一起判定
      if (quality->first != 1) {
        FLOWENGINE_LOGGER_INFO("Face quality is not good, type is {}!",
                               quality->first);
        cv::imwrite("test_liveness_quality_failed_out.jpg", image_bgr);
        continue;
      } else {
        isFaceReady = true;
        continue; // 后面开始动作判定
      }
    }

    // 如果状态为点头或摇头，则不需要进行关键点检测，直接用检测算法的关键点即可
    if (actionType == LivenessStatus::NOD ||
        actionType == LivenessStatus::SHAKE) {
      // 计算角度
      float pitch, yaw, roll;
      utils::points5angle(kbbox.points, pitch, yaw, roll);
      FLOWENGINE_LOGGER_DEBUG("pitch: {}, yaw: {}, roll: {}", pitch, yaw, roll);

      if (actionType == LivenessStatus::NOD) {
        FLOWENGINE_LOGGER_CRITICAL("Please nod your head!");
        if (pitch < 40) {
          cv::imwrite("test_liveness_nod_failed_out.jpg", image_bgr);
          continue;
        }
      } else {
        if (yaw < 40) {
          FLOWENGINE_LOGGER_CRITICAL("Please shake your head!");
          cv::imwrite("test_liveness_shake_failed_out.jpg", image_bgr);
          continue;
        }
      }
      // 到此处意味着点头或摇头成功，重置状态
      successCount++;
      actionStatus = false;
      actionType = LivenessStatus::None;
      cv::imwrite("test_liveness_out.jpg", image_bgr);
      continue;
    }

    // 到此意味着需要进行关键点检测后进行眨眼或张嘴判定
    float w = kbbox.bbox.bbox[2] - kbbox.bbox.bbox[0];
    float h = kbbox.bbox.bbox[3] - kbbox.bbox.bbox[1];
    cv::Point2f center =
        cv::Point2f{kbbox.bbox.bbox[0] + (w / 2), kbbox.bbox.bbox[1] + (h / 2)};
    // 旋转矩阵
    int rotation = 0;
    float maxSize = std::max(w, h) * 1.5;
    float scale = KEY_POINTS_INPUT_SIZE / maxSize;

    cv::Mat rotMat = transform(KEY_POINTS_INPUT_SIZE, center, scale, rotation);
    std::cout << "rotMat: " << rotMat << std::endl;

    cv::Mat aligned_face_bgr =
        normCrop(image_bgr, rotMat, KEY_POINTS_INPUT_SIZE);
    // std::string prefix = std::to_string(rand() % 1000);
    // cv::imwrite(prefix + "_liveness_aligned_face.jpg", aligned_face_bgr);

    cv::Mat aligned_face_nv12;
    utils::BGR2NV12(aligned_face_bgr, aligned_face_nv12);

    // 人脸关键点检测
    FrameInfo faceInput;
    faceInput.data = reinterpret_cast<void **>(&aligned_face_nv12.data);
    faceInput.inputShape = {aligned_face_nv12.cols, aligned_face_nv12.rows,
                            aligned_face_nv12.channels()};
    faceInput.shape = {aligned_face_nv12.cols, aligned_face_nv12.rows * 2 / 3,
                       aligned_face_nv12.channels()};
    faceInput.type = ColorType::NV12;
    InferResult facePointsRet;
    facePointsRet.aRet =
        KeypointsRet{.points = {}, .M = reinterpret_cast<float *>(rotMat.data)};
    auto ret = algoManager.infer(AlgoType::KEYPOINT, faceInput, facePointsRet);
    if (!ret.get()) {
      FLOWENGINE_LOGGER_INFO("Face keypoints detection failed!");
      continue;
    }
    auto keyPoints = std::get_if<KeypointsRet>(&facePointsRet.aRet);

    if (!keyPoints) {
      FLOWENGINE_LOGGER_INFO("Face keypoints detection failed!");
      continue;
    }

    if (actionType == LivenessStatus::MOUTH) {
      FLOWENGINE_LOGGER_CRITICAL("Please open your mouth!");
      // 计算嘴巴张开程度
      float ratio = mouthAspectRatio(keyPoints->points);
      FLOWENGINE_LOGGER_DEBUG("mouth aspect ratio: {}", ratio);
      if (ratio < 0.7) {
        cv::imwrite("test_liveness_mouth_failed_out.jpg", image_bgr);
        continue;
      }
    } else {
      FLOWENGINE_LOGGER_CRITICAL("Please blink your eyes!");
      // 计算眨眼程度
      float ratio = eyeAspectRatio(keyPoints->points);
      FLOWENGINE_LOGGER_DEBUG("eye aspect ratio: {}", ratio);
      if (ratio < 0.2) {
        cv::imwrite("test_liveness_eye_failed_out.jpg", image_bgr);
        continue;
      }
    }
    // 到此处意味着眨眼或张嘴成功，重置状态
    successCount++;
    actionStatus = false;
    actionType = LivenessStatus::None;

    {
      // 可视化人脸关键点
      for (auto &p : keyPoints->points) {
        cv::circle(image_bgr,
                   cv::Point{static_cast<int>(p.x), static_cast<int>(p.y)}, 2,
                   cv::Scalar{255, 255, 0});
      }
      cv::imwrite("test_liveness_out.jpg", image_bgr);
    }
  }

  gflags::ShutDownCommandLineFlags();
  return 0;
}
