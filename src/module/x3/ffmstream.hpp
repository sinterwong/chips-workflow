/**
 * @file ffmstream.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-30
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __STREAM_MANAGER_FOR_FFMPAGE_H_
#define __STREAM_MANAGER_FOR_FFMPAGE_H_
#include "logger/logger.hpp"
#include <ostream>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include "libavcodec/avcodec.h"
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#ifdef __cplusplus
}
#endif /* __cplusplus */

namespace module::utils {

class FmpStream {

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

  // ffmepg data packet
  AVPacket avpacket = {0};

  AvParam av_param;

  std::atomic_bool isOpen = false;

  std::string uri;

  static const std::unordered_map<AVCodecID, std::string> codecMapping;

public:
  bool openStream(); // 开启视频流

  int getRawFrame(void *data);

  inline bool isRunning() { return isOpen.load(); };

  inline int getWidth() {
    if (isRunning()) {
      return static_cast<int>(
          avContext->streams[av_param.videoIndex]->codecpar->width);
    } else {
      FLOWENGINE_LOGGER_ERROR("The stream is not opened!");
      return 0;
    }
  }
  inline int getHeight() {
    if (isRunning()) {
      return static_cast<int>(
          avContext->streams[av_param.videoIndex]->codecpar->height);
    } else {
      FLOWENGINE_LOGGER_ERROR("The stream is not opened!");
      return 0;
    }
  }

  inline std::string getCodecType() {

    if (isRunning()) {
      auto pCodec = avcodec_find_decoder(
          avContext->streams[av_param.videoIndex]->codecpar->codec_id);
      // FLOWENGINE_LOGGER_CRITICAL("Look here for codec_type: {}",
      // avContext->streams[av_param.videoIndex]->codecpar->codec_type);
      // FLOWENGINE_LOGGER_CRITICAL("Look here codec id: {}", pCodec->id);
      // FLOWENGINE_LOGGER_CRITICAL("Look here codec: {}",
      // codecMapping.at(pCodec->id));
      return codecMapping.at(pCodec->id);
    } else {
      FLOWENGINE_LOGGER_ERROR("The stream is not opened!");
      return 0;
    }
  }

  inline AvParam &getParam() { return av_param; }

  inline void closeStream() {

    auto lv = &avpacket;
    av_packet_free(&lv);

    if (avContext) {
      avformat_close_input(&avContext);
    }
    isOpen.store(false);
  }

  explicit FmpStream(std::string const &uri_) noexcept : uri(uri_) {}

  ~FmpStream() noexcept { closeStream(); }
};
} // namespace module::utils

#endif