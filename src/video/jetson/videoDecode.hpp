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
#ifndef __FLOWENGINE_JETSON_VIDEO_DECODE_H_
#define __FLOWENGINE_JETSON_VIDEO_DECODE_H_
#include "common/common.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include "videoSource.h"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <shared_mutex>
#include <thread>

#include "vdecoder.hpp"

namespace video {

class VideoDecode : private VDecoder {
private:
  std::shared_mutex stream_m;
  std::unique_ptr<videoSource> stream = nullptr;
  std::unique_ptr<joining_thread> consumer; // 消费者
  std::mutex frame_m;
  uchar3 *frame = nullptr;
  static const std::unordered_map<std::string, std::string> codecMapping;

  void consumeFrame();

  void fillOptionByCV(std::string const &url, videoOptions &options);

public:
  bool init() override;

  bool start(std::string const &, int w = 0, int h = 0) override;

  bool stop() override;

  inline bool isRunning() override {
    std::shared_lock lk{stream_m};
    return stream && stream->IsStreaming();
  }

  inline int getHeight() override {
    std::shared_lock lk{stream_m};
    if (isRunning()) {
      return stream->GetHeight();
    }
    return -1;
  }

  inline int getWidth() override {
    std::shared_lock lk{stream_m};
    if (isRunning()) {
      return stream->GetWidth();
    }
    return -1;
  }

  inline int getRate() override {
    std::shared_lock lk{stream_m};
    if (isRunning()) {
      return stream->GetFrameRate();
    }
    return -1;
  }

  std::shared_ptr<cv::Mat> getcvImage() override;

  inline common::ColorType getType() const noexcept override {
    return common::getPlatformColorType();
  }

  explicit VideoDecode() {}

  ~VideoDecode() noexcept {}
};
} // namespace video

#endif