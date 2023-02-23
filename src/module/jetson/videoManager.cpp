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
#include "opencv2/videoio.hpp"
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace module::utils {

bool VideoManager::init() {
  auto video = cv::VideoCapture();
  video.open(uri);
  opt.height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
  opt.width = video.get(cv::CAP_PROP_FRAME_WIDTH);
  opt.frameRate = video.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(video.get(cv::CAP_PROP_FOURCC));
  opt.codec = videoOptions::CodecFromStr(getCodec(fourcc).c_str());
  opt.resource = uri;
  stream = std::unique_ptr<videoSource>(videoSource::Create(opt));
  if (!stream) {
    FLOWENGINE_LOGGER_ERROR("jetson source:  failed to create input stream");
    return false;
  }
  video.release();
  // 读取一帧之后，视频流进入streaming状态
  bool ret = stream->Open();
  return true;
}

void VideoManager::streamGet() {

  // 每隔100ms消耗一帧，防止长时间静止
  while (isRunning()) {
    // FLOWENGINE_LOGGER_CRITICAL("Get Frame!");
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::lock_guard lk(m);
    uchar3 *temp = nullptr;
    stream->Capture(&temp, 1000);
  }
}

bool VideoManager::run() {
  if (!isRunning()) {
    return false;
  }
  consumer = std::make_unique<std::thread>(&VideoManager::streamGet, this);
  return true;
}

std::shared_ptr<cv::Mat> VideoManager::getcvImage() {
  // std::lock_guard lk(m);
  // if (!frame) {
  //   return std::shared_ptr<cv::Mat>();
  // }
  // return std::make_shared<cv::Mat>(cv::Mat(stream->GetHeight(),
  //                                          stream->GetWidth(), CV_8UC3,
  //                                          reinterpret_cast<void *>(frame)));

  std::lock_guard lk(m);
  bool ret = stream->Capture(&frame, 1000);
  if (!ret) {
    return nullptr;
  }
  return std::make_shared<cv::Mat>(cv::Mat(stream->GetHeight(),
                                           stream->GetWidth(), CV_8UC3,
                                           reinterpret_cast<void *>(frame)));
}

} // namespace module::utils