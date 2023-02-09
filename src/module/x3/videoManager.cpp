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
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace module::utils {

bool VideoManager::init() {
  videoOptions opt;
  // TODO global variable to update the channel number;
  opt.videoIdx = videoId % 32;
  opt.resource = uri;
  stream = videoSource::create(opt);
  FLOWENGINE_LOGGER_INFO("video initialization is successed!");
  return true;
}

bool VideoManager::run() {
  if (!stream->open()) {
    return false;
  };
  joining_thread consumer{[this]() {
    while (isRunning()) {
      std::lock_guard lk(m);
      bool ret = stream->capture(&frame, 1000);
      if (!ret) {
        frame = nullptr;
      }
    }
    FLOWENGINE_LOGGER_INFO("streamGet is over!");
  }};
  consumer.detach();
  return true;
}

cv::Mat VideoManager::getcvImage() {
  std::lock_guard lk(m);
  if (!frame) {
    return cv::Mat();
  }
  return cv::Mat(getHeight() * 3 / 2, getWidth(), CV_8UC1, frame).clone();
}

} // namespace module::utils