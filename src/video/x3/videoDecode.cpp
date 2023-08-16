/**
 * @file videoDecode.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-11
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "videoDecode.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "opencv2/videoio.hpp"
#include "video_utils.hpp"

using namespace std::chrono_literals;

namespace video {

bool VideoDecode::init() {
  videoOptions opt;
  opt.videoIdx = channel;
  opt.resource = uri;
  opt.height = h;
  opt.width = w;
  stream = videoSource::Create(opt);
  return true;
}

void VideoDecode::consumeFrame() {
  while (isRunning()) {
    // 每隔100ms消耗一帧，防止长时间静止
    {
      std::lock_guard lk(frame_m);
      bool ret = stream->Capture(&frame, 1000);
      if (!ret) {
        FLOWENGINE_LOGGER_WARN("Getframe is failed!");
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

bool VideoDecode::run() {
  if (!stream->Open()) {
    return false;
  }
  consumer = std::make_unique<joining_thread>(&VideoDecode::consumeFrame, this);
  return true;
}

std::shared_ptr<cv::Mat> VideoDecode::getcvImage() {
  std::lock_guard lk(frame_m);
  bool ret = stream->Capture(&frame, 1000);
  if (!ret) {
    FLOWENGINE_LOGGER_WARN("{} Getframe is failed!", stream->GetResource().string);
    return nullptr;
  }
  /** TODO
   * @brief frame管理优化 
   * 设置一个frame buffer，缓解刷新过快的问题 
   */
  return std::make_shared<cv::Mat>(
      cv::Mat(getHeight() * 3 / 2, getWidth(), CV_8UC1, frame).clone());
}
} // namespace video