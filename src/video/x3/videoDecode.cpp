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
  // if (uri.empty()) {
  //   // X3这里在构造对象的时候需要初始化解码器
  //   stream = videoSource::Create();
  // } else {
  //   videoOptions opt;
  //   opt.videoIdx = channel;
  //   opt.resource = uri;
  //   opt.height = h;
  //   opt.width = w;
  //   stream = videoSource::Create(opt);
  // }
  // return stream.get();

  // 如果uri还没有初始化init时什么也不做
  if (!uri.empty()) {
    videoOptions opt;
    opt.videoIdx = channel;
    opt.resource = uri;
    opt.height = h;
    opt.width = w;
    stream = videoSource::Create(opt);
  }
  return true;
}

bool VideoDecode::start(const std::string &url) {
  if (stream && stream->IsStreaming()) {
    FLOWENGINE_LOGGER_INFO("The stream had started {}",
                           stream->GetResource().string);
    return false;
  }
  uri = url;
  videoOptions opt;
  opt.videoIdx = channel;
  opt.resource = uri;

  // return stream->Open(opt);
  // TODO 每次start都是重新创建对象，因为目前来看x3的decoder不是很稳定
  stream = videoSource::Create(opt);
  return stream->Open();
}

bool VideoDecode::stop() {
  if (!stream || !stream->IsStreaming()) {
    FLOWENGINE_LOGGER_ERROR("The stream was not started {}.", uri);
    return false;
  }
  stream->Close();
  // TODO 安全起见，将对象一并释放掉重新创建了
  stream.reset();
  return true;
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
      bool ret = stream->Capture(&frame, 1000);
      if (!ret) {
        FLOWENGINE_LOGGER_WARN("Getframe is failed!");
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

} // namespace video