/**
 * @file videoManager.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-12-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __VIDEO_MANAGER_FOR_X3_H_
#define __VIDEO_MANAGER_FOR_X3_H_
#include "common/common.hpp"
#include "logger/logger.hpp"
#include "videoSource.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <thread>

namespace module::utils {

class VideoManager : private common::NonCopyable {
private:
  std::string uri; // 流地址
  int videoId;     // 编码通道
  std::unique_ptr<videoSource> stream;
  void* frame;

public:
  bool init();

  bool run();

  inline bool isRunning() { return stream->isStreaming(); }

  inline int getHeight() { return stream->getHeight(); }

  inline int getWidth() { return stream->getWidth(); }

  inline int getRate() { return stream->getFrameRate(); }

  cv::Mat getcvImage();

  inline common::ColorType getType() const noexcept {
    return stream->getRawFormat();
  }

  explicit VideoManager(std::string const &uri_, int idx_) noexcept
      : uri(uri_), videoId(idx_) {}

  ~VideoManager() noexcept {}
};
} // namespace module::utils

#endif