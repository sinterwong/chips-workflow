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
#ifndef __VIDEO_MANAGER_FOR_JETSON_H_
#define __VIDEO_MANAGER_FOR_JETSON_H_
#include "common/common.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include "videoOptions.h"
#include "videoSource.h"
#include <memory>
#include <opencv2/core/mat.hpp>

namespace module::utils {

class VideoManager : private common::NonCopyable {
private:
  // stream manager
  std::string uri;  // 流地址
  int videoId;      // 编码通道
  int mmzIndex;     // 循环索引
  videoOptions opt; // 视频参数
  std::unique_ptr<videoSource> stream = nullptr;
  uchar3 *frame;
  std::unique_ptr<std::thread> consumer; // 消费者
  std::mutex m;

  inline std::string getCodec(int fourcc) {
    char a[5];
    for (int i = 0; i < 4; i++) {
      a[i] = fourcc >> (i * 8) & 255;
    }
    a[4] = '\0';
    return std::string{a};
  }

  void streamGet();

public:
  bool init();

  void run();

  inline bool isRunning() { return stream && stream->IsStreaming(); }

  inline int getHeight() { return stream && stream->GetHeight(); }

  inline int getWidth() { return stream && stream->GetWidth(); }

  inline int getRate() { return stream && stream->GetFrameRate(); }

  inline void join() noexcept { consumer->join(); }

  cv::Mat getcvImage();

  inline common::ColorType getType() const noexcept {
    return common::ColorType::RGB888;
  }

  explicit VideoManager(std::string const &uri_, int idx_) noexcept
      : uri(uri_), videoId(idx_) {}

  ~VideoManager() noexcept { join(); }
};
} // namespace module::utils

#endif