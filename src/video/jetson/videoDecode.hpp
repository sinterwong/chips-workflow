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
#include "videoOptions.h"
#include "videoSource.h"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <thread>

#include "vdecoder.hpp"

namespace video {

class VideoDecode : private VDecoder {
private:
  std::string uri; // 流地址
  int w, h;        // 流分辨率
  std::unique_ptr<videoSource> stream;
  std::unique_ptr<joining_thread> consumer; // 消费者
  std::mutex frame_m;
  uchar3 *frame = nullptr;

  void consumeFrame();

public:
  bool init();

  bool run();

  inline bool isRunning() { return stream && stream->IsStreaming(); }

  inline int getHeight() {
    if (isRunning()) {
      return stream->GetHeight();
    }
    return -1;
  }

  inline int getWidth() {
    if (isRunning()) {
      return stream->GetWidth();
    }
    return -1;
  }

  inline int getRate() {
    if (isRunning()) {
      return stream->GetFrameRate();
    }
    return -1;
  }

  std::shared_ptr<cv::Mat> getcvImage();

  inline common::ColorType getType() const noexcept {
    return common::ColorType::RGB888;
  }

  explicit VideoDecode(std::string const &uri_, int w_ = 1920, int h_ = 1080)
      : uri(uri_), w(w_), h(h_) {}

  ~VideoDecode() noexcept {}
};
} // namespace video

#endif