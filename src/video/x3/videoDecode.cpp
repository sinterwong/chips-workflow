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
#include "common/joining_thread.h"
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
  // init 只用于初始化解码器资源，如果包装了解码器，这里就不需要做任何事情
  return true;
}

bool VideoDecode::start(const std::string &uri, int w, int h) {
  if (stream && stream->IsStreaming()) {
    FLOWENGINE_LOGGER_INFO("The stream had started {}",
                           stream->GetResource().string);
    return false;
  }
  videoOptions opt;
  opt.videoIdx = channel;
  opt.resource = uri;

  stream = videoSource::Create(opt);

  if (!stream->Open()) {
    FLOWENGINE_LOGGER_ERROR("Open stream failed!");
    return false;
  }
  consumer = std::make_unique<joining_thread>(&VideoDecode::consumeFrame, this);
  FLOWENGINE_LOGGER_INFO("The stream had started {}",
                         stream->GetResource().string);
  return true;
}

bool VideoDecode::stop() {
  if (!stream || !stream->IsStreaming()) {
    FLOWENGINE_LOGGER_ERROR("There is no stream running!");
    return false;
  }
  stream->Close();
  // TODO 安全起见，将对象一并释放掉重新创建了
  stream.reset();
  return true;
}

std::shared_ptr<cv::Mat> VideoDecode::getcvImage() {
  std::lock_guard lk(frame_m);
  bool ret = stream->Capture(&frame, 1000);
  if (!ret) {
    FLOWENGINE_LOGGER_WARN("{} Getframe is failed!",
                           stream->GetResource().string);
    return nullptr;
  }
  /** TODO
   * @brief frame管理优化
   * 设置一个frame buffer，缓解刷新过快的问题
   */
  return std::make_shared<cv::Mat>(
      cv::Mat(getHeight() * 3 / 2, getWidth(), CV_8UC1, frame).clone());
}

void VideoDecode::consumeFrame() {
  while (isRunning()) {
    // 每隔一段时间消耗一帧，防止长时间静止造成 ffmpeg 连接失效
    {
      std::lock_guard lk(frame_m);
      if (!isRunning()) {
        break;
      }
      bool ret = stream->Capture(&frame, 1000);
      if (!ret) {
        FLOWENGINE_LOGGER_WARN("Getframe is failed!");
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

} // namespace video