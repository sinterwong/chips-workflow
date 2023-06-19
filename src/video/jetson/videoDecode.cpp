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

std::unordered_map<std::string, std::string> const VideoDecode::codecMapping =
    {std::make_pair("h264", "h264"), std::make_pair("h265", "h265"),
     std::make_pair("avc", "h264"), std::make_pair("hevc", "h265")};

bool VideoDecode::init() {

  // 利用opencv打开视频，获取配置
  videoOptions opt;
  auto video = cv::VideoCapture();
  video.open(uri);
  opt.height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
  opt.width = video.get(cv::CAP_PROP_FRAME_WIDTH);
  opt.frameRate = video.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(video.get(cv::CAP_PROP_FOURCC));
  std::string scodec = codecMapping.at(utils::getCodec(fourcc));
  opt.codec = videoOptions::CodecFromStr(scodec.c_str());
  opt.resource = uri;
  stream = std::unique_ptr<videoSource>(videoSource::Create(opt));
  video.release();
  return true;
}

void VideoDecode::consumeFrame() {
  while (isRunning()) {
    // 每隔100ms消耗一帧，防止长时间静止
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    std::lock_guard lk(frame_m);
    bool ret = stream->Capture(&frame, 1000);
    if (!ret) {
      FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    }
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
    FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    return nullptr;
  }
  return std::make_shared<cv::Mat>(cv::Mat(stream->GetHeight(),
                                           stream->GetWidth(), CV_8UC3,
                                           reinterpret_cast<void *>(frame)));
}

} // namespace video