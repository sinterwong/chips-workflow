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
#include "vdecoder.hpp"

namespace video {

class VideoDecode : private VDecoder {
private:
  std::string uri;
  int w, h;
  void consumeFrame();

public:
  bool init() override;

  bool run() override;

  inline bool isRunning() override { return true; }

  inline int getHeight() override { return -1; }

  inline int getWidth() override { return -1; }

  inline int getRate() override { return -1; }

  std::shared_ptr<cv::Mat> getcvImage() override;

  inline common::ColorType getType() const noexcept override {
    return common::ColorType::NV12;
  }

  explicit VideoDecode(std::string const &uri_, int w_ = 1920, int h_ = 1080)
      : uri(uri_), w(w_), h(h_) {}

  ~VideoDecode() noexcept {}
};
} // namespace video

#endif