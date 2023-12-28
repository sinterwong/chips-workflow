/**
 * @file videoDecode.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-11
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __FLOWENGINE_VIDEO_DECODE_H_
#define __FLOWENGINE_VIDEO_DECODE_H_
#include "common/common.hpp"
#include "common/joining_thread.h"
#include "logger/logger.hpp"
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <shared_mutex>
#include <thread>

#include "videoSource.hpp"
#include "video_common.hpp"

#include "vdecoder.hpp"

namespace video {

class VideoDecode : private VDecoder {
private:
  std::shared_mutex stream_m;
  std::unique_ptr<videoSource> stream;
  std::unique_ptr<joining_thread> consumer; // 消费者
  std::mutex frame_m;
  void *frame = nullptr;
  int channel;

  void consumeFrame();

public:
  bool init() override;

  bool start(std::string const &, int w = 0, int h = 0) override;

  bool stop() override;

  inline bool isRunning() override {
    std::shared_lock lk(stream_m);
    return stream && stream->IsStreaming();
  }

  inline int getHeight() override {
    std::shared_lock lk(stream_m);
    if (isRunning()) {
      return stream->GetHeight();
    }
    return -1;
  }

  inline int getWidth() override {
    std::shared_lock lk(stream_m);
    if (isRunning()) {
      return stream->GetWidth();
    }
    return -1;
  }

  inline int getRate() override {
    std::shared_lock lk(stream_m);
    if (isRunning()) {
      return stream->GetFrameRate();
    }
    return -1;
  }

  std::shared_ptr<cv::Mat> getcvImage() override;

  inline common::ColorType getType() const noexcept override {
    return common::getPlatformColorType();
  }

  explicit VideoDecode() {
    channel = ChannelsManager::getInstance().getChannel();
    if (channel < 0) {
      throw std::runtime_error("Channel usage overflow!");
    }
  }

  ~VideoDecode() noexcept {
    if (isRunning()) {
      stop();
    }
    // 释放channel
    ChannelsManager::getInstance().setChannel(channel);
  }
};
} // namespace video

#endif