/**
 * @file ffstream.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-09
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "ffstream.hpp"
#include "logger/logger.hpp"
#include <cstdlib>

#include <mutex>

using namespace std::chrono_literals;

#define SET_BYTE(_p, _b) *_p++ = (unsigned char)_b;

#define SET_BUFFER(_p, _buf, _len)                                             \
  memcpy(_p, _buf, _len);                                                      \
  (_p) += (_len);

namespace video::utils {

std::unordered_map<AVCodecID, std::string> const FFStream::codecMapping = {
    std::make_pair(AV_CODEC_ID_NONE, "none"),
    std::make_pair(AV_CODEC_ID_H264, "h264"),
    std::make_pair(AV_CODEC_ID_H265, "h265")};

static int build_dec_seq_header(uint8_t *pbHeader, std::string const &p_enType,
                                const AVStream *st, int *sizelength) {

  AVCodecParameters *avc = st->codecpar;

  uint8_t *pbMetaData = avc->extradata;
  int nMetaData = avc->extradata_size;
  uint8_t *p = pbMetaData;
  uint8_t *a = p + 4 - ((long)p & 3);
  uint8_t *t = pbHeader;
  int size;
  int sps, pps, i, nal;

  size = 0;
  *sizelength = 4; // default size length(in bytes) = 4
  if (p_enType == "h264") {
    if (nMetaData > 1 && pbMetaData && pbMetaData[0] == 0x01) {
      // check mov/mo4 file format stream
      p += 4;
      *sizelength = (*p++ & 0x3) + 1;
      sps = (*p & 0x1f); // Number of sps
      p++;
      for (i = 0; i < sps; i++) {
        nal = (*p << 8) + *(p + 1) + 2;
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x01);
        SET_BUFFER(t, p + 2, nal - 2);
        p += nal;
        size += (nal - 2 + 4); // 4 => length of start code to be inserted
      }

      pps = *(p++); // number of pps
      for (i = 0; i < pps; i++) {
        nal = (*p << 8) + *(p + 1) + 2;
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x00);
        SET_BYTE(t, 0x01);
        SET_BUFFER(t, p + 2, nal - 2);
        p += nal;
        size += (nal - 2 + 4); // 4 => length of start code to be inserted
      }
    } else if (nMetaData > 3) {
      size = -1; // return to meaning of invalid stream data;
      for (; p < a; p++) {
        if (p[0] == 0 && p[1] == 0 && p[2] == 1) {
          // find startcode
          size = avc->extradata_size;
          if (!pbHeader || !pbMetaData)
            return 0;
          SET_BUFFER(pbHeader, pbMetaData, size);
          break;
        }
      }
    }
  } else if (p_enType == "h265") {
    if (nMetaData > 1 && pbMetaData && pbMetaData[0] == 0x01) {
      static const int8_t nalu_header[4] = {0, 0, 0, 1};
      int numOfArrays = 0;
      uint16_t numNalus = 0;
      uint16_t nalUnitLength = 0;
      uint32_t offset = 0;

      p += 21;
      *sizelength = (*p++ & 0x3) + 1;
      numOfArrays = *p++;

      while (numOfArrays--) {
        p++; // NAL type
        numNalus = (*p << 8) + *(p + 1);
        p += 2;
        for (i = 0; i < numNalus; i++) {
          nalUnitLength = (*p << 8) + *(p + 1);
          p += 2;
          // if(i == 0)
          {
            memcpy(pbHeader + offset, nalu_header, 4);
            offset += 4;
            memcpy(pbHeader + offset, p, nalUnitLength);
            offset += nalUnitLength;
          }
          p += nalUnitLength;
        }
      }

      size = offset;
    } else if (nMetaData > 3) {
      size = -1; // return to meaning of invalid stream data;

      for (; p < a; p++) {
        if (p[0] == 0 && p[1] == 0 && p[2] == 1) // find startcode
        {
          size = avc->extradata_size;
          if (!pbHeader || !pbMetaData)
            return 0;
          SET_BUFFER(pbHeader, pbMetaData, size);
          break;
        }
      }
    }
  } else {
    SET_BUFFER(pbHeader, pbMetaData, nMetaData);
    size = nMetaData;
  }

  return size;
}

bool FFStream::openStream(bool withCodec) {
  std::lock_guard<std::shared_mutex> lk(ctx_m);

  int ret = 0;
  AVDictionary *option = nullptr;
  av_dict_set(&option, "stimeout", "3000000", 0);
  av_dict_set(&option, "bufsize", "1024000", 0);
  av_dict_set(&option, "rtsp_transport", "tcp", 0);
  ret = avformat_open_input(&avContext, uri.c_str(), 0, &option);
  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("avformat_open_input failed {}", uri);
    av_dict_free(&option); // 释放 option 内存
    return false;
  }
  ret = avformat_find_stream_info(avContext, 0);
  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("avformat_find_stream_info failed {}", uri);
    return false;
  }
  FLOWENGINE_LOGGER_INFO("probesize: {}", avContext->probesize);

  /* dump input information to stderr */
  av_dump_format(avContext, 0, uri.c_str(), 0);
  av_param.videoIndex =
      av_find_best_stream(avContext, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (av_param.videoIndex < 0) {
    FLOWENGINE_LOGGER_ERROR("av_find_best_stream failed, ret: {}, url: {}",
                            av_param.videoIndex, uri);
    return false;
  }

  av_packet_unref(&avpacket);
  isOpen.store(true);

  av_param.firstPacket = 1;

  if (withCodec) {
    if (!openCodec()) {
      FLOWENGINE_LOGGER_ERROR("openCodec failed");
      return false;
    }
  }
  return true;
}

bool FFStream::openCodec() {
  if (!isRunning()) {
    FLOWENGINE_LOGGER_ERROR("The stream is not opened!");
    return false;
  }

  int ret = 0;

  AVCodecParameters *codecParameters =
      avContext->streams[av_param.videoIndex]->codecpar;
  AVCodec const *avCodec = avcodec_find_decoder(
      avContext->streams[av_param.videoIndex]->codecpar->codec_id);

  if (!avCodec) {
    FLOWENGINE_LOGGER_ERROR("avcodec_find_decoder failed");
    return false;
  }

  avCodecContext = avcodec_alloc_context3(avCodec);
  if (!avCodecContext) {
    FLOWENGINE_LOGGER_ERROR("avcodec_alloc_context3 failed");
    return false;
  }

  ret = avcodec_parameters_to_context(avCodecContext, codecParameters);
  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("avcodec_parameters_to_context failed");
    return false;
  }

  ret = avcodec_open2(avCodecContext, avCodec, nullptr);
  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("avcodec_open2 failed");
    return false;
  }

  if (!frame) {
    frame = av_frame_alloc();
    if (!frame) {
      FLOWENGINE_LOGGER_ERROR("av_frame_alloc failed");
      return false;
    }
  }

  if (!frame_rgb) {
    frame_rgb = av_frame_alloc();
    if (!frame_rgb) {
      FLOWENGINE_LOGGER_ERROR("av_frame_alloc failed");
      return false;
    }
  }
  rgbBufSize = av_image_get_buffer_size(AV_PIX_FMT_RGB24, avCodecContext->width,
                                        avCodecContext->height, 1);
  rgbBuf = (uint8_t *)av_malloc(rgbBufSize * sizeof(uint8_t));
  av_image_fill_arrays(frame_rgb->data, frame_rgb->linesize, rgbBuf,
                       AV_PIX_FMT_RGB24, avCodecContext->width,
                       avCodecContext->height, 1);

  // 转换像素格式为RGB24
  swsCtx = sws_getContext(avCodecContext->width, avCodecContext->height,
                          avCodecContext->pix_fmt, avCodecContext->width,
                          avCodecContext->height, AV_PIX_FMT_RGB24,
                          SWS_BILINEAR, nullptr, nullptr, nullptr);

  if (!swsCtx) {
    FLOWENGINE_LOGGER_ERROR("sws_getContext failed");
    return false;
  }

  FLOWENGINE_LOGGER_INFO("open codec success! codec: {}", avCodec->name);
  return true;
}

int FFStream::handleDecode() {
  int ret = avcodec_send_packet(avCodecContext, &avpacket);
  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("avcodec_send_packet failed, ret: {}", ret);
    return 0;
  }
  ret = avcodec_receive_frame(avCodecContext, frame);
  if (ret < 0) {
    FLOWENGINE_LOGGER_ERROR("avcodec_receive_frame failed, ret: {}", ret);
    return 0;
  }

  sws_scale(swsCtx, frame->data, frame->linesize, 0, avCodecContext->height,
            frame_rgb->data, frame_rgb->linesize);
  return frame_rgb->linesize[0] * frame->height;
}

int FFStream::handleFirstPacket(void **data, bool isCopy) {
  int seqHeaderSize = 0;
  seqHeaderSize = 0;
  AVCodecParameters *codec;
  int retSize = 0;
  codec = avContext->streams[av_param.videoIndex]->codecpar;
  if (seqHeader) { // 如果此时seqHeader已经申请过内存，需要先释放
    free(seqHeader);
    seqHeader = nullptr;
  }
  seqHeader = (uint8_t *)malloc(codec->extradata_size + 1024);
  if (seqHeader == nullptr) {
    FLOWENGINE_LOGGER_INFO("Failed to mallock seqHeader");
    return -1;
  }
  memset((void *)seqHeader, 0x00, codec->extradata_size + 1024);

  seqHeaderSize = build_dec_seq_header(
      seqHeader, codecMapping.at(avContext->video_codec_id),
      avContext->streams[av_param.videoIndex], &retSize);
  if (seqHeaderSize < 0) {
    FLOWENGINE_LOGGER_INFO("Failed to build seqHeader");
    return -1;
  }
  av_param.firstPacket = 0;
  av_param.bufSize = seqHeaderSize;

  void *source;
  if (avCodecContext) {
    av_param.bufSize = handleDecode();
    if (av_param.bufSize < 0) {
      return -1;
    }
    source = (void *)frame_rgb->data[0];
  } else {
    source = (void *)seqHeader;
    av_param.bufSize = seqHeaderSize;
  }

  if (isCopy) {
    memcpy(*data, source, av_param.bufSize);
  } else {
    *data = source;
  }
  return av_param.bufSize;
}

int FFStream::handleSubsequentPackets(void **data, bool isCopy,
                                      bool onlyIFrame) {
  if (onlyIFrame) {
    if (!(avpacket.flags & AV_PKT_FLAG_KEY)) {
      // 这不是I帧，所以我们释放数据包并返回
      av_packet_unref(&avpacket);
      return 0;
    }
  }

  void *source;
  if (avCodecContext) {
    av_param.bufSize = handleDecode();
    if (av_param.bufSize < 0) {
      return -1;
    }
    source = (void *)frame_rgb->data[0];
  } else {
    source = (void *)avpacket.data;
    av_param.bufSize = avpacket.size;
  }

  if (isCopy) {
    memcpy(*data, source, av_param.bufSize);
  } else {
    *data = source;
  }
  FLOWENGINE_LOGGER_DEBUG("av_param.bufSize: {}", av_param.bufSize);
  return av_param.bufSize;
}

int FFStream::getDataFrame(void **data, bool isCopy, bool onlyIFrame) {
  std::lock_guard<std::shared_mutex> lk(ctx_m);
  if (!isRunning())
    return -1;

  av_packet_unref(&avpacket);
  int error = av_read_frame(avContext, &avpacket);
  if (error < 0) {
    handleReadFrameError(error); // 处理读取帧错误的函数
    return -1;
  }
  if (av_param.firstPacket) {
    // 处理第一个包的函数
    return handleFirstPacket(data, isCopy);
  } else {
    // 处理后续包的函数
    return handleSubsequentPackets(data, isCopy, onlyIFrame);
  }
}
} // namespace video::utils