/**
 * @file ffmstream.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-30
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "ffmstream.hpp"
#include "libavcodec/avcodec.h"

using namespace std::chrono_literals;

#define SET_BYTE(_p, _b) *_p++ = (unsigned char)_b;

#define SET_BUFFER(_p, _buf, _len)                                             \
  memcpy(_p, _buf, _len);                                                      \
  (_p) += (_len);

namespace module::utils {

std::unordered_map<AVCodecID, std::string> const FmpStream::codecMapping = {
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

bool FmpStream::openStream() {
  int ret = 0;

  AVDictionary *option = 0;
  av_dict_set(&option, "stimeout", "3000000", 0);
  av_dict_set(&option, "bufsize", "1024000", 0);
  av_dict_set(&option, "rtsp_transport", "tcp", 0);
  ret = avformat_open_input(&avContext, uri.c_str(), 0, &option);
  if (ret < 0) {
    FLOWENGINE_LOGGER_INFO("avformat_open_input failed");
    return false;
  }
  ret = avformat_find_stream_info(avContext, 0);
  if (ret < 0) {
    FLOWENGINE_LOGGER_INFO("avformat_find_stream_info failed");
    return false;
  }
  FLOWENGINE_LOGGER_INFO("probesize: {}", avContext->probesize);

  /* dump input information to stderr */
  av_dump_format(avContext, 0, uri.c_str(), 0);
  av_param.videoIndex =
      av_find_best_stream(avContext, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (av_param.videoIndex < 0) {
    FLOWENGINE_LOGGER_INFO("av_find_best_stream failed, ret: {}", av_param.videoIndex);
    return false;
  }
  av_init_packet(&avpacket);
  avpacket.data = nullptr;
  avpacket.size = 0;
  isOpen.store(true);

  av_param.firstPacket = 1;
  return true;
}

int FmpStream::getRawFrame(void *data) {
  uint8_t *seqHeader = nullptr;
  int seqHeaderSize = 0, error = 0;

  if (!avpacket.size) {
    error = av_read_frame(avContext, &avpacket);
  }
  if (error < 0) {
    if (error == AVERROR_EOF || avContext->pb->eof_reached) {
      FLOWENGINE_LOGGER_INFO("There is no more input data, {}!", avpacket.size);
    } else {
      FLOWENGINE_LOGGER_INFO("Failed to av_read_frame error(0x{})", error);
    }
    closeStream();
    return -1;
  } else {
    if (av_param.firstPacket) {
      seqHeaderSize = 0;
      AVCodecParameters *codec;
      int retSize = 0;
      codec = avContext->streams[av_param.videoIndex]->codecpar;
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
      memcpy(data, (void *)seqHeader, seqHeaderSize);
      if (seqHeader) {
        free(seqHeader);
        seqHeader = nullptr;
      }
    } else {
      av_param.bufSize = avpacket.size;
      memcpy((void *)data, (void *)avpacket.data, avpacket.size);
      av_packet_unref(&avpacket);
      avpacket.size = 0;
    }
    ++av_param.count;
    return av_param.bufSize;
  }
}

} // namespace module::utils