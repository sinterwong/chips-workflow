/**
 * @file faceQuality.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "faceQuality.hpp"
#include "algoManager.hpp"
#include "logger/logger.hpp"
#include "networkUtils.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <tuple>

using namespace infer;

using namespace common;

namespace server::face::core {

FaceQuality *FaceQuality::instance = nullptr;

bool FaceQuality::infer(std::string const &url, int &quality) {
  // 推理任务定义
  std::shared_ptr<cv::Mat> image_bgr = getImageByUri(url);
  if (image_bgr == nullptr) {
    return false;
  }

  // TODO: 暂时写死NV12格式推理
  cv::Mat input;
  infer::utils::BGR2NV12(*image_bgr, input);
  return infer(FramePackage{url, std::make_shared<cv::Mat>(input)}, quality);
}

bool FaceQuality::infer(FramePackage const &framePackage, int &quality) {

  // 使用算法资源进行推理
  FrameInfo frame;
  frame.data = reinterpret_cast<void **>(&framePackage.frame->data);
  frame.inputShape = {framePackage.frame->cols, framePackage.frame->rows,
                      framePackage.frame->channels()};

  // TODO:暂时默认是NV12格式，这里应该有一个宏来确定是什么推理数据
  frame.shape = {framePackage.frame->cols, framePackage.frame->rows * 2 / 3,
                 framePackage.frame->channels()};
  frame.type = ColorType::NV12;

  // 人脸检测
  InferResult faceDetRet;
  auto ret = AlgoManager::getInstance().infer(AlgoType::DET, frame, faceDetRet);
  if (!ret.get()) {
    FLOWENGINE_LOGGER_INFO("Face detection failed!");
    return false;
  }
  auto kbboxes = std::get_if<KeypointsBoxes>(&faceDetRet.aRet);
  if (!kbboxes || kbboxes->empty()) {
    FLOWENGINE_LOGGER_INFO("Not a single face was detected!");
    return false;
  }

  // 获取最靠近中心的人脸
  size_t index = utils::findClosestBBoxIndex(*kbboxes, framePackage.frame->cols,
                                             framePackage.frame->rows);

  auto kbbox = kbboxes->at(index);

  // 根据检测结果判定人脸是否合格
  float faceWidth = kbbox.bbox.bbox[2] - kbbox.bbox.bbox[0];
  float faceHeight = kbbox.bbox.bbox[3] - kbbox.bbox.bbox[1];
  // 1. 图像尺寸
  if (faceWidth < 80 || faceHeight < 80) {
    quality = 1;
    FLOWENGINE_LOGGER_DEBUG("faceWidth: {}, faceHeight: {}", faceWidth,
                            faceHeight);
    return true;
  }

  // 2. 长宽比
  if (faceWidth / faceHeight > 2.5 || faceHeight / faceWidth > 2.5) {
    quality = 2;
    FLOWENGINE_LOGGER_DEBUG("faceWidth: {}, faceHeight: {}", faceWidth,
                            faceHeight);
    return true;
  }

  // 获取人脸识别输入图像
  FrameInfo faceInput;
  cv::Mat faceImage;
  if (!getFaceInput(*framePackage.frame, faceImage, faceInput, kbbox.bbox,
                    frame.type)) {
    FLOWENGINE_LOGGER_ERROR("Get face input failed!");
    return false;
  }

  // 3. 亮度和锐度
  // 默认图像是NV12格式
  cv::Mat faceImageBGR;
  cv::cvtColor(faceImage, faceImageBGR, cv::COLOR_YUV2BGR_NV12);
  float sharpness, brightness;
  std::tie(sharpness, brightness) =
      utils::sharpnessAndBrightnessScore(faceImageBGR);

  if (sharpness <= 0.4 || brightness <= 0.4) {
    quality = 3;
    FLOWENGINE_LOGGER_DEBUG("sharpness: {}, brightness: {}", sharpness,
                            brightness);
    return true;
  }

  // 4. 角度
  float pitch, yaw, roll;
  utils::points5angle(kbbox.points, pitch, yaw, roll);
  // pitch 为仰头角度，yaw 为左右转头角度，roll 为摇头角度
  if (pitch >= 30 || yaw >= 35 || roll >= 30) {
    quality = 4;
    FLOWENGINE_LOGGER_DEBUG("pitch: {}, yaw: {}, roll: {}", pitch, yaw, roll);
    return true;
  }

  // 5. 遮挡
  InferResult faceQualityRet;
  ret = AlgoManager::getInstance().infer(AlgoType::QUALITY, faceInput,
                                         faceQualityRet);
  if (!ret.get()) {
    FLOWENGINE_LOGGER_INFO("Face quality failed!");
    return false;
  }
  auto aRet = std::get_if<ClsRet>(&faceQualityRet.aRet);
  if (!aRet) {
    FLOWENGINE_LOGGER_INFO("Face quality failed!");
    return false;
  }
  int cls = aRet->first;
  if (cls == 0) {
    quality = 5; // 大胡子
    return true;
  } else if (cls == 1) {
    quality = 0; // 正常
    return true;
  } else if (cls == 2) {
    quality = 6; // 眼镜
    return true;
  } else if (cls == 3) {
    quality = 7; // 口罩
    return true;
  } else if (cls == 4) {
    quality = 8; // 墨镜
    return true;
  } else if (cls == 5) {
    quality = 9; // 遮挡
    return true;
  }
  return true;
}

bool FaceQuality::getFaceInput(cv::Mat const &input, cv::Mat &output,
                               FrameInfo &frame, common::BBox const &bbox,
                               ColorType const &type) {
  int x, y, w, h;
  x = static_cast<int>(bbox.bbox[0]);
  y = static_cast<int>(bbox.bbox[1]);
  w = static_cast<int>(bbox.bbox[2]) - x;
  h = static_cast<int>(bbox.bbox[3]) - y;

  cv::Rect retBox{x, y, w, h};
  if (!utils::cropImage(input, output, retBox, type)) {
    FLOWENGINE_LOGGER_ERROR("Crop image failed!");
    return false;
  }

  frame.data = reinterpret_cast<void **>(&output.data);
  // 输入的图像尺寸
  frame.inputShape = {output.cols, output.rows, output.channels()};
  frame.type = type;

  if (type == ColorType::NV12) {
    // RGB图像的尺寸
    frame.shape = {output.cols, output.rows * 2 / 3, output.channels()};
  } else {
    frame.shape = {output.cols, output.rows, output.channels()};
  }
  return true;
}
} // namespace server::face::core