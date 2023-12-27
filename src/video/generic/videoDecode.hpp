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

#ifndef __FLOWENGINE_GENERIC_VIDEO_DECODE_H_
#define __FLOWENGINE_GENERIC_VIDEO_DECODE_H_
#include "common/common.hpp"
#include "common/joining_thread.h"
#include "ffstream.hpp"
#include "logger/logger.hpp"
#include <memory>
#include <mutex>
#include <thread>

#include "video_common.hpp"

#include "vdecoder.hpp"

namespace video {

class VideoDecode : private VDecoder {
private:
  std::unique_ptr<FFStream> stream;
  std::unique_ptr<joining_thread> consumer; // 消费者
  std::mutex frame_m;

  void consumeFrame();

public:
  bool init() override;

  bool start(std::string const &, int w = 0, int h = 0) override;

  bool stop() override;

  inline bool isRunning() override { return stream && stream->isRunning(); }

  inline int getHeight() override {
    if (isRunning()) {
      return stream->getHeight();
    }
    return -1;
  }

  inline int getWidth() override {
    if (isRunning()) {
      return stream->getWidth();
    }
    return -1;
  }

  inline int getRate() override {
    if (isRunning()) {
      return stream->getRate();
    }
    return -1;
  }

  std::shared_ptr<cv::Mat> getcvImage() override;

  inline common::ColorType getType() const noexcept override {
    return common::getPlatformColorType();
  }

  explicit VideoDecode() {}

  ~VideoDecode() noexcept {
    std::lock_guard<std::mutex> lock(frame_m);
    stream.reset();
    consumer->join();
  }
};
} // namespace video

#endif