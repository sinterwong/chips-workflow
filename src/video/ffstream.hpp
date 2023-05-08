/**
 * @file ffstream.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-09
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __STREAM_MANAGER_FFMPAGE_H_
#define __STREAM_MANAGER_FFMPAGE_H_
#include "libavutil/rational.h"
#include "logger/logger.hpp"
#include <mutex>
#include <ostream>
#include <shared_mutex>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include "libavcodec/avcodec.h"
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#ifdef __cplusplus
}
#endif /* __cplusplus */

namespace video::utils {

class FFStream {

  struct AvParam {
    int count = 0;
    int videoIndex;
    int bufSize;
    int firstPacket;

    friend std::ostream &operator<<(std::ostream &os, AvParam &param) {
      os << "count: " << param.count << ", "
         << "videoIndex: " << param.videoIndex << ", "
         << "bufSize: " << param.bufSize << ", "
         << "firstPacket: " << param.firstPacket << std::endl;
      return os;
    }
  };

private:
  // ffmpeg context
  AVFormatContext *avContext = nullptr;
  std::shared_mutex ctx_m;

  // ffmepg data packet
  AVPacket avpacket = {0};

  AvParam av_param;

  std::atomic_bool isOpen = false;

  std::string uri;

  static const std::unordered_map<AVCodecID, std::string> codecMapping;

  uint8_t *seqHeader = nullptr;

public:
  bool openStream(); // 开启视频流

  int getRawFrame(void **data);

  inline bool isRunning() { return isOpen.load(); };

  inline int getWidth() {
    std::shared_lock<std::shared_mutex> lk(ctx_m);
    if (isRunning()) {
      return static_cast<int>(
          avContext->streams[av_param.videoIndex]->codecpar->width);
    } else {
      FLOWENGINE_LOGGER_ERROR("The stream is not opened!");
      return 0;
    }
  }
  
  inline int getHeight() {
    std::shared_lock<std::shared_mutex> lk(ctx_m);
    if (isRunning()) {
      return static_cast<int>(
          avContext->streams[av_param.videoIndex]->codecpar->height);
    } else {
      FLOWENGINE_LOGGER_ERROR("The stream is not opened!");
      return 0;
    }
  }

  inline int getRate() {
    std::shared_lock<std::shared_mutex> lk(ctx_m);
    if (isRunning()) {
      return static_cast<int>(
          av_q2d(avContext->streams[av_param.videoIndex]->r_frame_rate));
    } else {
      FLOWENGINE_LOGGER_ERROR("The stream is not opened!");
      return 0;
    }
  }

  inline std::string getCodecType() {
    std::shared_lock<std::shared_mutex> lk(ctx_m);
    if (isRunning()) {
      auto pCodec = avcodec_find_decoder(
          avContext->streams[av_param.videoIndex]->codecpar->codec_id);
      return codecMapping.at(pCodec->id);
    } else {
      FLOWENGINE_LOGGER_ERROR("The stream is not opened!");
      return 0;
    }
  }

  inline AvParam &getParam() { return av_param; }

  inline void closeStream() {
    std::lock_guard lk(ctx_m);
    if (avContext) {
      avformat_close_input(&avContext);
    }
    isOpen.store(false);
  }

  explicit FFStream(std::string const &uri_) noexcept : uri(uri_) {}

  ~FFStream() noexcept {
    if (isRunning()) {
      closeStream();
    }
    if (seqHeader) {
      free(seqHeader);
      seqHeader = nullptr;
    }
  }
};
} // namespace module::utils

#endif