/**
 * @file StreamGenerator.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-10-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_SUNRISE_DECODER_H
#define __METAENGINE_SUNRISE_DECODER_H

#include <any>
#include <memory>
#include <vector>

#include "common/common.hpp"
#include "common/config.hpp"
#include "logger/logger.hpp"
#include "messageBus.h"
#include "module.hpp"

#include "x3_sdk_wrap.hpp"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
#include <libavformat/avformat.h>
#include <libavutil/timestamp.h>
#ifdef __cplusplus
}
#endif /* __cplusplus */
namespace module {

class StreamGenerator : public Module {
private:
  // ffmpeg context
  AVFormatContext *avContext = nullptr;

  // ffmepg data packet
  AVPacket avpacket = {0};

  // 用于存储码流数据（内部模式直接把码流的buffer地址给到vir_ptr字段就行）
  VIDEO_STREAM_S pstStream;

  // 获取解码之后的数据
  VIDEO_FRAME_S stFrameInfo;

  // 帧数索引
  int mmz_index = 0;
  // 流拉取状态
  int error;
  
  int ret = 0;

  // x3 info
  x3_vdec_chn_info_t vdec_chn_info;

  CameraResult cameraResult;

private:
  int vdec_ChnAttr_init(VDEC_CHN_ATTR_S *pVdecChnAttr, PAYLOAD_TYPE_E enType,
                        int picWidth, int picHeight) {
    // int streambufSize = 0;
    if (pVdecChnAttr == NULL) {
      printf("pVdecChnAttr is NULL!\n");
      return -1;
    }
    // 该步骤必不可少
    memset(pVdecChnAttr, 0, sizeof(VDEC_CHN_ATTR_S));
    // 设置解码模式分别为 PT_H264 PT_H265 PT_MJPEG
    pVdecChnAttr->enType = enType;
    // 设置解码模式为帧模式
    pVdecChnAttr->enMode = VIDEO_MODE_FRAME;
    // 设置像素格式 NV12格式
    pVdecChnAttr->enPixelFormat = HB_PIXEL_FORMAT_NV12;
    // 输入buffer个数
    pVdecChnAttr->u32FrameBufCnt = 3;
    // 输出buffer个数
    pVdecChnAttr->u32StreamBufCnt = 3;
    // 输出buffer size，必须1024对齐
    pVdecChnAttr->u32StreamBufSize =
        (picWidth * picHeight * 3 / 2 + 1024) & ~0x3ff;
    // 使用外部buffer
    pVdecChnAttr->bExternalBitStreamBuf = HB_TRUE;
    if (enType == PT_H265) {
      // 使能带宽优化
      pVdecChnAttr->stAttrH265.bandwidth_Opt = HB_TRUE;
      // 普通解码模式，IPB帧解码
      pVdecChnAttr->stAttrH265.enDecMode = VIDEO_DEC_MODE_NORMAL;
      // 输出顺序按照播放顺序输出
      pVdecChnAttr->stAttrH265.enOutputOrder = VIDEO_OUTPUT_ORDER_DISP;
      // 不启用CLA作为BLA处理
      pVdecChnAttr->stAttrH265.cra_as_bla = HB_FALSE;
      // 配置tempral id为绝对模式
      pVdecChnAttr->stAttrH265.dec_temporal_id_mode = 0;
      // 保持2
      pVdecChnAttr->stAttrH265.target_dec_temporal_id_plus1 = 2;
    }
    if (enType == PT_H264) {
      // 使能带宽优化
      pVdecChnAttr->stAttrH264.bandwidth_Opt = HB_TRUE;
      // 普通解码模式，IPB帧解码
      pVdecChnAttr->stAttrH264.enDecMode = VIDEO_DEC_MODE_NORMAL;
      // 输出顺序按照播放顺序输出
      pVdecChnAttr->stAttrH264.enOutputOrder = VIDEO_OUTPUT_ORDER_DISP;
    }
    if (enType == PT_JPEG) {
      // 使用内部buffer
      pVdecChnAttr->bExternalBitStreamBuf = HB_FALSE;
      // 配置镜像模式，不镜像
      pVdecChnAttr->stAttrJpeg.enMirrorFlip = DIRECTION_NONE;
      // 配置旋转模式，不旋转
      pVdecChnAttr->stAttrJpeg.enRotation = CODEC_ROTATION_0;
      // 配置crop，不启用
      pVdecChnAttr->stAttrJpeg.stCropCfg.bEnable = HB_FALSE;
    }
    return 0;
  }

public:
  StreamGenerator(Backend *ptr, const std::string &initName,
                       const std::string &initType,
                       const common::CameraConfig &_params,
                       
                       );

  ~StreamGenerator() {
    if (avContext)
      avformat_close_input(&avContext);
  }

  virtual void forward(std::vector<forwardMessage> message) override;

  void step() override;

  static void delBuffer(std::vector<std::any> &);

  // std::any getFrameInfo(std::vector<std::any> &, FrameBuf *);

  // std::any getMatBuffer(std::vector<std::any> &list, FrameBuf *buf);

  // std::any getPtrBuffer(std::vector<std::any> &list, FrameBuf *buf);
};
} // namespace module
#endif // __METAENGINE_JETSON_SOURCE_H
