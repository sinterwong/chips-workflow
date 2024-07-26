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
#include "ffstream.hpp"
#include <mutex>
#include <opencv2/imgproc.hpp>
#include <shared_mutex>
#include <thread>

using namespace std::chrono_literals;

namespace video {

bool VideoDecode::init() {
  // init 只用于初始化解码器资源，如果包装了解码器，这里就不需要做任何事情
  return true;
}

bool VideoDecode::start(const std::string &uri, int w, int h) {
  std::lock_guard lk{stream_m};
  if (stream && stream->isRunning()) {
    FLOWENGINE_LOGGER_INFO("The stream had started {}", uri);
    return false;
  }
  stream = std::make_unique<FFStream>(uri);
  if (!stream->openStream(true)) {
    FLOWENGINE_LOGGER_ERROR("Can't open stream {}", uri);
    return false;
  }
  consumer =
      std::make_unique<utils::joining_thread>(&VideoDecode::consumeFrame, this);
  FLOWENGINE_LOGGER_INFO("The stream had started {}", uri);
  return true;
}

bool VideoDecode::stop() {
  std::lock_guard lk{stream_m};
  if (stream && stream->isRunning()) {
    stream->closeStream();
  }
  stream.reset();
  return true;
}

std::shared_ptr<cv::Mat> VideoDecode::getcvImage() {
  std::shared_lock lk{stream_m};
  void *data = nullptr;
  int bufSize = stream->getDataFrame(&data);
  if (bufSize < 0) {
    FLOWENGINE_LOGGER_ERROR("VideoDecode::getcvImage() UNKNOWN FAILED.");
    throw std::runtime_error("UNKNOWN FAILED.");
  } else if (bufSize == 0) {
    FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    return nullptr;
  }
  // 数据需要拷贝出去，因为流中并没有拷贝，因此获取下一帧时会覆盖当前帧
  return std::make_shared<cv::Mat>(
      cv::Mat{stream->getHeight(), stream->getWidth(), CV_8UC3, data}.clone());
}

void VideoDecode::consumeFrame() {
  while (isRunning()) {
    // 每隔100ms消耗一帧，防止长时间静止
    {
      std::shared_lock lk{stream_m};
      void *tempData = nullptr;
      int bufSize = stream->getDataFrame(&tempData);
      if (bufSize < 0) {
        FLOWENGINE_LOGGER_ERROR("VideoDecode::consumeFrame() UNKNOWN FAILED.");
        throw std::runtime_error("UNKNOWN FAILED.");
      } else if (bufSize == 0) {
        // 当前帧失败
        FLOWENGINE_LOGGER_WARN("current frame failed.");
      }
    }
    std::this_thread::sleep_for(100ms);
  }
}

} // namespace video