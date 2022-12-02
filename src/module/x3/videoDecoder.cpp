/**
 * @file videoDecoder.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-25
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "videoDecoder.hpp"
#include "logger/logger.hpp"
#include "x3_vio_vp.hpp"
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>

using namespace std::chrono_literals;

namespace module::utils {

const static bool isInit = []() -> bool {
  // 编码、解码模块初始化，整个应用中需要调用一次
  HB_VDEC_Module_Init();
  auto ret = x3_vp_init();
  if (ret) {
    HB_VDEC_Module_Uninit();
    return false;
  }
  FLOWENGINE_LOGGER_INFO("x3_vp_init ok!");
  return true;
}();

std::unordered_map<std::string, PAYLOAD_TYPE_E> const
    VideoDecoder::entypeMapping = {std::make_pair("h264", PT_H264),
                                   std::make_pair("h265", PT_H265)};

bool VideoDecoder::init(std::string const &type_, int width_, int height_) {
  assert(isInit);
  type = type_;
  width = width_;
  height = height_;
  if (!vdec_ChnAttr_init(m_chn_attr, width, height, entypeMapping.at(type))) {
    FLOWENGINE_LOGGER_ERROR("vdec_ChnAttr_init failed, {}");
    return false;
  }

  // 初始化解码器
  if (!x3_vdec_init(m_vdec_chn_id, &m_chn_attr)) {
    FLOWENGINE_LOGGER_ERROR("x3_vdec_init failed");
    return false;
  }
  if (!x3_vdec_start(m_vdec_chn_id)) {
    FLOWENGINE_LOGGER_ERROR("x3_vdec_start failed");
    return false;
  }
  FLOWENGINE_LOGGER_INFO("start vdec chn{} ok!", m_vdec_chn_id);

  // 初始化frame包装器
  memset(&pstStream, 0, sizeof(VIDEO_STREAM_S));

  return true;
}

bool VideoDecoder::vdec_ChnAttr_init(VDEC_CHN_ATTR_S &pVdecChnAttr, int width,
                                     int height,
                                     PAYLOAD_TYPE_E enType) noexcept {
  // int streambufSize = 0;
  // 该步骤必不可少
  memset(&pVdecChnAttr, 0, sizeof(VDEC_CHN_ATTR_S));
  // 设置解码模式分别为 PT_H264 PT_H265 PT_MJPEG
  pVdecChnAttr.enType = enType;
  // 设置解码模式为帧模式
  pVdecChnAttr.enMode = VIDEO_MODE_FRAME;
  // 设置像素格式 NV12格式
  pVdecChnAttr.enPixelFormat = HB_PIXEL_FORMAT_NV12;
  // 输入buffer个数
  pVdecChnAttr.u32FrameBufCnt = 3;
  // 输出buffer个数
  pVdecChnAttr.u32StreamBufCnt = 3;
  // 输出buffer size，必须1024对齐
  pVdecChnAttr.u32StreamBufSize = (width * height * 3 / 2 + 1024) & ~0x3ff;
  // 使用外部buffer
  pVdecChnAttr.bExternalBitStreamBuf = HB_TRUE;
  if (enType == PT_H265) {
    // 使能带宽优化
    pVdecChnAttr.stAttrH265.bandwidth_Opt = HB_TRUE;
    // 普通解码模式，IPB帧解码
    pVdecChnAttr.stAttrH265.enDecMode = VIDEO_DEC_MODE_NORMAL;
    // 输出顺序按照播放顺序输出
    pVdecChnAttr.stAttrH265.enOutputOrder = VIDEO_OUTPUT_ORDER_DISP;
    // 不启用CLA作为BLA处理
    pVdecChnAttr.stAttrH265.cra_as_bla = HB_FALSE;
    // 配置tempral id为绝对模式
    pVdecChnAttr.stAttrH265.dec_temporal_id_mode = 0;
    // 保持2
    pVdecChnAttr.stAttrH265.target_dec_temporal_id_plus1 = 2;
  }
  if (enType == PT_H264) {
    // 使能带宽优化
    pVdecChnAttr.stAttrH264.bandwidth_Opt = HB_TRUE;
    // 普通解码模式，IPB帧解码
    pVdecChnAttr.stAttrH264.enDecMode = VIDEO_DEC_MODE_NORMAL;
    // 输出顺序按照播放顺序输出
    pVdecChnAttr.stAttrH264.enOutputOrder = VIDEO_OUTPUT_ORDER_DISP;
  }
  if (enType == PT_JPEG) {
    // 使用内部buffer
    pVdecChnAttr.bExternalBitStreamBuf = HB_FALSE;
    // 配置镜像模式，不镜像
    pVdecChnAttr.stAttrJpeg.enMirrorFlip = DIRECTION_NONE;
    // 配置旋转模式，不旋转
    pVdecChnAttr.stAttrJpeg.enRotation = CODEC_ROTATION_0;
    // 配置crop，不启用
    pVdecChnAttr.stAttrJpeg.stCropCfg.bEnable = HB_FALSE;
  }
  return true;
}

bool VideoDecoder::sendStream(int index, int count, uint64_t paddr, char *vaddr,
                              int size) {
  int error;
  VDEC_CHN_STATUS_S pstStatus;
  HB_VDEC_QueryStatus(m_vdec_chn_id, &pstStatus);
  if (pstStatus.cur_input_buf_cnt >= maxCnt) {
    std::this_thread::sleep_for(10ms);
  }
  pstStream.pstPack.phy_ptr = paddr;
  pstStream.pstPack.vir_ptr = vaddr;
  pstStream.pstPack.pts = count;
  pstStream.pstPack.src_idx = index;
  pstStream.pstPack.size = size;
  pstStream.pstPack.stream_end = HB_FALSE;

  error = HB_VDEC_SendStream(m_vdec_chn_id, &pstStream, 3000);
  if (error == -HB_ERR_VDEC_OPERATION_NOT_ALLOWDED ||
      error == -HB_ERR_VDEC_UNKNOWN) {
    FLOWENGINE_LOGGER_ERROR("sendStream: HB_VDEC_SendStream failed");
    return false;
  }
  return true;
}

// std::ostream &operator<<(std::ostream &os, VIDEO_FRAME_S &frame) {

//   os << frame.stVFrame.width << ", " << frame.stVFrame.height << ", "
//      << frame.stVFrame.size << ", " << frame.stVFrame.vir_ptr << ", "
//      << frame.stVFrame.vir_ptr[0] << ", " << frame.stVFrame.src_idx << ", "
//      << frame.stVFrame.frame_end << ", " << frame.stVFrame.pts << std::endl;

//   return os;
// }

bool VideoDecoder::getFrame(VIDEO_FRAME_S &stFrameInfo) {
  // std::cout << stFrameInfo << std::endl;
  int error = HB_VDEC_GetFrame(m_vdec_chn_id, &stFrameInfo, 1000);
  if (error) {
    FLOWENGINE_LOGGER_ERROR("HB_VDEC_GetFrame chn{} error, ret: {}",
                            m_vdec_chn_id, error);
    return false;
  }
  return true;
}

} // namespace module::utils