/**
 * @file videoDecode.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "videoDecode.hpp"
#include <opencv2/imgproc.hpp>

using namespace std::chrono_literals;

namespace video {

bool VideoDecode::init() {
  // 如果uri还没有初始化Jetson在init时什么也不做
  if (!uri.empty()) {
    stream = std::make_unique<cv::VideoCapture>(uri);
    if (!isRunning()) {
      FLOWENGINE_LOGGER_ERROR("Can't open stream {}", uri);
      return false;
    }
  }
  return true;
}

bool VideoDecode::start(const std::string &url) {

  if (stream && stream->isOpened()) {
    FLOWENGINE_LOGGER_INFO("The stream had started {}", url);
    return false;
  }
  uri = url;
  stream = std::make_unique<cv::VideoCapture>(uri);
  if (!stream->isOpened()) {
    FLOWENGINE_LOGGER_ERROR("Can't open stream {}", uri);
    return false;
  }
  return true;
}

bool VideoDecode::stop() {
  if (stream && stream->isOpened()) {
    stream->release();
  }
  stream.reset();
  return true;
}

bool VideoDecode::run() {
  if (!isRunning()) {
    FLOWENGINE_LOGGER_ERROR("The stream had not started {}", uri);
    return false;
  }
  consumer = std::make_unique<joining_thread>([this]() { consumeFrame(); });
  return true;
}

std::shared_ptr<cv::Mat> VideoDecode::getcvImage() {
  std::lock_guard<std::mutex> lock(frame_m);
  cv::Mat frame;
  if (!stream->read(frame)) {
    FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    return nullptr;
  }
  cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
  return std::make_shared<cv::Mat>(frame);
}

void VideoDecode::consumeFrame() {
  while (isRunning()) {
    // 每隔100ms消耗一帧，防止长时间静止
    std::this_thread::sleep_for(100ms);
    std::lock_guard<std::mutex> lock(frame_m);
    cv::Mat frame;
    if (!stream->read(frame)) {
      FLOWENGINE_LOGGER_WARN("Getframe is failed!");
      continue;
    }
  }
}

} // namespace video