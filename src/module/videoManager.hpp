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
#ifndef __FLOWENGINE_VIDEO_MANAGER_H_
#define __FLOWENGINE_VIDEO_MANAGER_H_
#include "common/common.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <thread>

#if (TARGET_PLATFORM == 0)
#include "x3/videoSource.hpp"
#include "x3/video_common.hpp"
#elif (TARGET_PLATFORM == 1)
#include "videoOptions.h"
#include "videoSource.h"
#elif (TARGET_PLATFORM == 2)
#endif

namespace module::utils {

class VideoManager : private common::NonCopyable {
private:
  std::string uri; // 流地址
  std::unique_ptr<videoSource> stream;
  std::unique_ptr<joining_thread> consumer; // 消费者
  std::mutex m;

#if (TARGET_PLATFORM == 0)
  void *frame = nullptr;
  int channel;
#elif (TARGET_PLATFORM == 1)
  uchar3 *frame = nullptr;
  int mmzIndex; // 循环索引
#endif

  void consumeFrame();

public:
  bool init();

  bool run();

  inline bool isRunning() { return stream && stream->IsStreaming(); }

  inline int getHeight() {
    if (stream) {
      return stream->GetHeight();
    }
    return -1;
  }

  inline int getWidth() {
    if (stream) {
      return stream->GetWidth();
    }
    return -1;
  }

  inline int getRate() {
    if (stream) {
      return stream->GetFrameRate();
    }
    return -1;
  }

  std::shared_ptr<cv::Mat> getcvImage();

  inline common::ColorType getType() const noexcept {
#if (TARGET_PLATFORM == 0)
    return common::ColorType::NV12;
#elif (TARGET_PLATFORM == 1)
    return common::ColorType::RGB888;
#endif
  }

  explicit VideoManager(std::string const &uri_) : uri(uri_) {
#if (TARGET_PLATFORM == 0)
    channel = ChannelsManager::getInstance().getChannel();
    if (channel < 0) {
      throw std::runtime_error("Channel usage overflow!");
    }
#endif
  }

  ~VideoManager() noexcept {
#if (TARGET_PLATFORM == 0)
    ChannelsManager::getInstance().setChannel(channel); // 返还channel
#endif
  }
};
} // namespace module::utils

#endif