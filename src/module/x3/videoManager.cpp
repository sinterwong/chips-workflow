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
#include "videoDecoder.hpp"
#include "videoSource.hpp"
#include "video_common.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace module::utils {

bool VideoManager::init() {
  videoOptions opt;
  opt.videoIdx = channel;
  opt.resource = uri;
  stream = videoSource::create(opt);
  FLOWENGINE_LOGGER_INFO("video initialization is successed, channel {}!", channel);
  return true;
}

void VideoManager::streamGet() {
  while (isRunning()) {
    // 每隔100ms消耗一帧，防止长时间静止
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::lock_guard lk(m);
    bool ret = stream->capture(&frame, 1000);
    if (!ret) {
      FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    }
  }
}

bool VideoManager::run() {
  if (!stream->open()) {
    return false;
  };
  consumer = std::make_unique<joining_thread>(&VideoManager::streamGet, this);
  return true;
}

// 可以返回一个shared_ptr
std::shared_ptr<cv::Mat> VideoManager::getcvImage() {
  std::lock_guard lk(m);
  if (!frame) {
    return nullptr;
  }
  bool ret = stream->capture(&frame, 1000);
  if (!ret) {
    FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    return nullptr;
  }
  return std::make_shared<cv::Mat>(
      cv::Mat(getHeight() * 3 / 2, getWidth(), CV_8UC1, frame));
}

} // namespace module::utils