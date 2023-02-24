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
#include "joining_thread.h"
#include "logger/logger.hpp"
#include "videoSource.hpp"
#include "video_common.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <thread>

namespace module::utils {

class VideoManager : private common::NonCopyable {
private:
  std::string uri; // 流地址
  int channel;
  std::unique_ptr<videoSource> stream;
  void *frame = nullptr;
  std::unique_ptr<joining_thread> consumer; // 消费者
  std::mutex m;

  void streamGet();

public:
  bool init();

  bool run();

  inline bool isRunning() { return stream && stream->isStreaming(); }

  inline int getHeight() {
    if (stream) {
      return stream->getHeight();
    }
    return -1;
  }

  inline int getWidth() {
    if (stream) {
      return stream->getWidth();
    }
    return -1;
  }

  inline int getRate() {
    if (stream) {
      return stream->getFrameRate();
    }
    return -1;
  }

  std::shared_ptr<cv::Mat> getcvImage();

  inline common::ColorType getType() const noexcept {
    return stream->getRawFormat();
  }

  explicit VideoManager(std::string const &uri_) : uri(uri_) {
    channel = ChannelsManager::getInstance().getChannel();
    if (channel < 0) {
      throw std::runtime_error("Channel usage overflow!");
    }
  }

  ~VideoManager() noexcept {
    ChannelsManager::getInstance().setChannel(channel);
  }
};
} // namespace module::utils

#endif