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
#include "ffstream.hpp"
#include "logger/logger.hpp"
#include "utils/joining_thread.hpp"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>

#include "video_common.hpp"

#include "vdecoder.hpp"

namespace video {

class VideoDecode : private VDecoder {
private:
  std::shared_mutex stream_m;
  std::unique_ptr<FFStream> stream;
  std::unique_ptr<utils::joining_thread> consumer; // 消费者

  void consumeFrame();

public:
  bool init() override;

  bool start(std::string const &, int w = 0, int h = 0) override;

  bool stop() override;

  inline bool isRunning() override {
    std::shared_lock lk{stream_m};
    return stream && stream->isRunning();
  }

  inline int getHeight() override {
    std::shared_lock lk{stream_m};
    if (isRunning()) {
      return stream->getHeight();
    }
    return -1;
  }

  inline int getWidth() override {
    std::shared_lock lk{stream_m};
    if (isRunning()) {
      return stream->getWidth();
    }
    return -1;
  }

  inline int getRate() override {
    std::shared_lock lk{stream_m};
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
    if (isRunning()) {
      stop();
    }
  }
};
} // namespace video

#endif