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
  std::string uri; // 流地址
  std::unique_ptr<FFStream> stream;
  std::unique_ptr<joining_thread> consumer; // 消费者
  std::mutex frame_m;
  int channel;

  void consumeFrame();

public:
  bool init() override;

  bool run() override;

  bool start(std::string const &) override;

  bool stop() override;

  inline bool isRunning() override { return stream && stream->isRunning(); }

  inline std::string getUri() override { return uri; }

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
    return common::ColorType::RGB888;
  }

  explicit VideoDecode() {}

  explicit VideoDecode(std::string const &uri_, int w_ = 1920, int h_ = 1080)
      : uri(uri_) {}

  ~VideoDecode() noexcept {}
};
} // namespace video

#endif