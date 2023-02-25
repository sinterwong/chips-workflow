/**
 * @file videoManager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-12-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "videoManager.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace module::utils {

bool VideoManager::init() {

#if (TARGET_PLATFORM == 0)
  videoOptions opt;
  opt.videoIdx = channel;
  opt.resource = uri;
  stream = videoSource::Create(opt);
#elif (TARGET_PLATFORM == 1)
  // 利用opencv打开视频，获取配置
  videoOptions opt;
  auto video = cv::VideoCapture();
  video.open(uri);
  opt.height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
  opt.width = video.get(cv::CAP_PROP_FRAME_WIDTH);
  opt.frameRate = video.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(video.get(cv::CAP_PROP_FOURCC));
  opt.codec = videoOptions::CodecFromStr(getCodec(fourcc).c_str());
  opt.resource = uri;
  stream = std::unique_ptr<videoSource>(videoSource::Create(opt));
  video.release();
#endif
  bool ret = stream->Open();
  if (!ret || !stream) {
    FLOWENGINE_LOGGER_ERROR(
        "VideoManager init:  failed to create input stream");
    return false;
  }
  FLOWENGINE_LOGGER_INFO(
      "VideoManager init: video initialization is successed!");
  return true;
}

void VideoManager::consumeFrame() {
  while (isRunning()) {
    // 每隔100ms消耗一帧，防止长时间静止
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::lock_guard lk(m);
    bool ret = stream->Capture(&frame, 1000);
    if (!ret) {
      FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    }
  }
}

bool VideoManager::run() {
  if (!stream->Open()) {
    return false;
  };
  consumer =
      std::make_unique<joining_thread>(&VideoManager::consumeFrame, this);
  return true;
}

std::shared_ptr<cv::Mat> VideoManager::getcvImage() {
  std::lock_guard lk(m);
  bool ret = stream->Capture(&frame, 1000);
  if (!ret) {
    FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    return nullptr;
  }
#if (TARGET_PLATFORM == 0)
  return std::make_shared<cv::Mat>(
      cv::Mat(getHeight() * 3 / 2, getWidth(), CV_8UC1, frame));
#elif (TARGET_PLATFORM == 1)
  return std::make_shared<cv::Mat>(cv::Mat(stream->GetHeight(),
                                           stream->GetWidth(), CV_8UC3,
                                           reinterpret_cast<void *>(frame)));
#endif
}

} // namespace module::utils