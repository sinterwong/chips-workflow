/**
 * @file videoDecoder.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-25
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __VIDEO_DECODER_FOR_X3_H_
#define __VIDEO_DECODER_FOR_X3_H_
#include "common/common.hpp"
#include "logger/logger.hpp"
#include <atomic>
#include <cstddef>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <unordered_map>

#include "hb_vdec.h"

#define VIDEO_CHECK_SUCCESS(value, errmsg)                                     \
  do {                                                                         \
    /*value can be call of function*/                                          \
    int ret_code = value;                                                      \
    if (ret_code != 0) {                                                       \
      FLOWENGINE_LOGGER_ERROR("[BPU ERROR] {}, error code:{}", errmsg,         \
                              ret_code);                                       \
      return false;                                                            \
    }                                                                          \
  } while (0);

namespace module::utils {

class VideoDecoder : private common::NonCopyable {
private:
  bool vdec_ChnAttr_init(VDEC_CHN_ATTR_S &pVdecChnAttr, int width, int height,
                         PAYLOAD_TYPE_E enType) noexcept;
  inline bool x3_vdec_init(VDEC_CHN vdecChn, VDEC_CHN_ATTR_S *vdecChnAttr) {
    VIDEO_CHECK_SUCCESS(HB_VDEC_CreateChn(vdecChn, vdecChnAttr),
                        "HB_VDEC_CreateChn failed");
    VIDEO_CHECK_SUCCESS(HB_VDEC_SetChnAttr(vdecChn, vdecChnAttr),
                        "HB_VDEC_SetChnAttr failed"); // config
    return true;
  }

  inline bool x3_vdec_deinit(VDEC_CHN vdecChn) {
    VIDEO_CHECK_SUCCESS(HB_VDEC_DestroyChn(vdecChn),
                        "HB_VDEC_ReleaseFrame failed");
    FLOWENGINE_LOGGER_INFO("ok!");
    return true;
  }

  inline bool x3_vdec_start(VDEC_CHN vdecChn) {
    VIDEO_CHECK_SUCCESS(HB_VDEC_StartRecvStream(vdecChn),
                        "HB_VDEC_StartRecvStream failed");
    isStart.store(true);
    return true;
  }

  inline bool x3_vdec_stop(VDEC_CHN vdecChn) {
    VIDEO_CHECK_SUCCESS(HB_VDEC_StopRecvStream(vdecChn),
                        "HB_VDEC_StopRecvStream failed");
    return true;
  }

private:
  // 用于存储码流数据（内部模式直接把码流的buffer地址给到vir_ptr字段就行）
  VIDEO_STREAM_S pstStream;

  // codec type
  std::string type;

  int width;

  int height;

  int m_vdec_chn_id; // 编码通道

  size_t maxCnt; // 最大缓存数量

  VDEC_CHN_ATTR_S m_chn_attr; // 通道属性

  std::atomic_bool isStart = false;

public:
  bool init(std::string const &type_, int width_, int height_);

  bool sendStream(int index, int count, uint64_t paddr, char *vaddr, int size);

  bool getFrame(VIDEO_FRAME_S &stFrameInfo);

  inline bool isRunning() { return isStart.load(); }

  inline common::ColorType getType() const noexcept {
    return common::ColorType::NV12;
  }

  explicit VideoDecoder(int idx_, size_t maxCnt_) noexcept
      : m_vdec_chn_id(idx_), maxCnt(maxCnt_){};

  ~VideoDecoder() noexcept {
    isStart.store(false);
    x3_vdec_stop(m_vdec_chn_id);
    x3_vdec_deinit(m_vdec_chn_id);
  }

  static const std::unordered_map<std::string, PAYLOAD_TYPE_E> entypeMapping;
};
} // namespace module::utils
#endif