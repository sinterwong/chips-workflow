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
namespace server::face::core {

ResultProcessor *ResultProcessor::instance = nullptr;

void ResultProcessor::onFrameReceived(FramePackage &&framePackage) {
  FLOWENGINE_LOGGER_DEBUG("Frame received and processed.");
  tpool->submit(
      [this, &framePackage]() { this->oneProcess(std::move(framePackage)); });
}

void ResultProcessor::oneProcess(FramePackage &&framePackage) {
  // 单个任务的函数
  // * 1. 根据帧包中的图片，送入人脸识别算法
  std::vector<float> feature;
  auto f = AlgoManager::getInstance().infer(framePackage, feature);
  if (!f.get()) {
    return;
  }

  // * 2. 人脸识别结果匹配人脸库
  auto idx = FaceLibraryManager::getInstance().match(feature.data(), 0.4);
  if (idx < 0) { // 没有匹配到人脸
    return;
  }

  // * 3. 根据人脸库返回的结果决定是否发送消息到后端服务
  PostInfo postInfo{framePackage.cameraName, idx};
  MY_CURL_POST(postUrl, {
    info["cameraName"] = postInfo.cameraName;
    info["id"] = postInfo.id;
  });
}
} // namespace server::face::core