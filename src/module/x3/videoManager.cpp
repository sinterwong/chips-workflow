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

bool VideoManager::run() { return stream->open(); }

cv::Mat VideoManager::getcvImage() {
  stream->capture(&frame);
  return cv::Mat(getHeight() * 3 / 2, getWidth(), CV_8UC1, frame).clone();
}

} // namespace module::utils