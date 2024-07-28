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
#include "joining_thread.hpp"
#include "logger/logger.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "opencv2/videoio.hpp"
#include "videoOptions.h"
#include "videoSource.h"
#include "video_utils.hpp"

using namespace std::chrono_literals;

namespace video {

std::unordered_map<std::string, std::string> const VideoDecode::codecMapping = {
    std::make_pair("h264", "h264"), std::make_pair("h265", "h265"),
    std::make_pair("avc", "h264"), std::make_pair("hevc", "h265")};

void VideoDecode::fillOptionByCV(std::string const &url, videoOptions &opt) {
  auto video = cv::VideoCapture();
  video.open(url, cv::CAP_FFMPEG);
  opt.height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
  opt.width = video.get(cv::CAP_PROP_FRAME_WIDTH);
  opt.frameRate = video.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(video.get(cv::CAP_PROP_FOURCC));
  std::string scodec = codecMapping.at(utils::getCodec(fourcc));
  opt.codec = videoOptions::CodecFromStr(scodec.c_str());
  opt.resource = url;
  video.release();
}

bool VideoDecode::init() {
  // init 只用于初始化解码器资源，如果包装了解码器，这里就不需要做任何事情
  return true;
}

bool VideoDecode::start(const std::string &url, int w, int h) {
  {
    std::lock_guard lk(stream_m);
    if (stream && stream->IsStreaming()) {
      FLOWENGINE_LOGGER_INFO("The stream had started {}",
                             stream->GetResource().string);
      return false;
    }
    videoOptions opt;
    fillOptionByCV(url, opt);
    stream = std::unique_ptr<videoSource>(videoSource::Create(opt));

    if (!stream) {
      FLOWENGINE_LOGGER_ERROR("Create stream failed!");
      return false;
    }

    if (!stream->Open()) {
      FLOWENGINE_LOGGER_ERROR("Open stream failed!");
      return false;
    }
  }

  consumer =
      std::make_unique<utils::joining_thread>(&VideoDecode::consumeFrame, this);
  FLOWENGINE_LOGGER_INFO("The stream had started {}",
                         stream->GetResource().string);
  return true;
}

bool VideoDecode::stop() {
  std::lock_guard lk(stream_m);
  if (!(stream && stream->IsStreaming())) {
    FLOWENGINE_LOGGER_ERROR("There is no stream running!");
    return false;
  }
  stream->Close();
  stream.reset();
  return true;
}

std::shared_ptr<cv::Mat> VideoDecode::getcvImage() {
  std::shared_lock lks(stream_m);
  std::lock_guard lk(frame_m);
  bool ret = stream->Capture(&frame, 1000);
  if (!ret) {
    FLOWENGINE_LOGGER_WARN("Getframe is failed!");
    return nullptr;
  }
  return std::make_shared<cv::Mat>(cv::Mat(stream->GetHeight(),
                                           stream->GetWidth(), CV_8UC3,
                                           reinterpret_cast<void *>(frame))
                                       .clone());
}

void VideoDecode::consumeFrame() {
  while (isRunning()) {
    {
      std::shared_lock lks(stream_m);
      std::lock_guard lk(frame_m);
      bool ret = stream->Capture(&frame, 1000);
      if (!ret) {
        FLOWENGINE_LOGGER_WARN("Getframe is failed!");
      }
    }
    // 每隔100ms消耗一帧，防止长时间静止
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

} // namespace video