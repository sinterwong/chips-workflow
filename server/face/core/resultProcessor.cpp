/**
 * @file resultProcessor.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "resultProcessor.hpp"
#include "logger/logger.hpp"
namespace server::face::core {

ResultProcessor *ResultProcessor::instance = nullptr;

void ResultProcessor::onFrameReceived(std::string const &lname,
                                      FramePackage framePackage,
                                      std::string const &postUrl) {
  tpool->submit([this, &lname, &postUrl, framePackage]() {
    this->oneProcess(lname, std::move(framePackage), postUrl);
  });
}

void ResultProcessor::oneProcess(std::string const &lname,
                                 FramePackage framePackage,
                                 std::string const &postUrl) {
  // 单个任务的函数
  // * 1. 根据帧包中的图片，送入人脸识别算法
  std::vector<float> feature;
  auto ret = FaceRecognition::getInstance().extract(framePackage, feature);
  if (!ret) {
    return;
  }

  // * 2. 人脸识别结果匹配人脸库
  auto idx =
      FaceLibraryManager::getInstance().match(lname, feature.data(), 0.4);
  if (idx < 0) { // 没有匹配到人脸
    return;
  }

  // * 3. 根据人脸库返回的结果决定是否发送消息到后端服务
  PostInfo postInfo{framePackage.cameraName, idx};
  MY_CURL_POST(postUrl, {
    info["camera_id"] = postInfo.cameraName;
    info["user_id"] = postInfo.id;
  });
}
} // namespace server::face::core