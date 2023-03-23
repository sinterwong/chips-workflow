/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef X3_SDK_WRAP_H_
#define X3_SDK_WRAP_H_

// SDK 提供的接口都经过这里封装好向上提供
#include "vio/hb_comm_vdec.h"
#include "vio/hb_comm_venc.h"
#include "vio/hb_common_vot.h"
#include "vio/hb_mipi_api.h"
#include "vio/hb_rgn.h"
#include "vio/hb_sys.h"
#include "vio/hb_vin_api.h"
#include "vio/hb_vio_interface.h"
#include "vio/hb_vps_api.h"
#include <string>

#define VIDEO_CHECK_SUCCESS(value, errmsg)                                        \
  do {                                                                         \
    /*value can be call of function*/                                          \
    int ret_code = value;                                                      \
    if (ret_code != 0) {                                                       \
      FLOWENGINE_LOGGER_ERROR("[BPU ERROR] {}, error code:{}", errmsg,          \
                             ret_code);                                        \
      return ret_code;                                                         \
    }                                                                          \
  } while (0);

#define SET_BYTE(_p, _b) *_p++ = (unsigned char)_b;

#define SET_BUFFER(_p, _buf, _len)                                             \
  memcpy(_p, _buf, _len);                                                      \
  (_p) += (_len);

struct av_param_t {
  int count;
  int videoIndex;
  int bufSize;
  int firstPacket;
};

// 以下两个结构体用来描述模块使用的内存buff信息
struct vp_param_t {
  uint64_t mmz_paddr[5];
  char *mmz_vaddr[5];
  int mmz_cnt = 5;
  int mmz_size;
};

// 单个解码通道的属性配置
struct x3_vdec_chn_info_t {
  // 解码数据源支持：
  // 1、 h264码流文件（已支持）
  // 2、 rtsp h264 码流
  std::string m_stream_src;
  // 解码器属性，内部通过union支持H264、H265、Mjpeg、Jpeg
  int m_vdec_chn_id; /* 编码通道 */
  int m_chn_enable;
  VDEC_CHN_ATTR_S m_chn_attr;

  vp_param_t vp_param;
  av_param_t av_param;
};

// 定义一组解码器，h264、h265最大32路编码器，jpeg、mjpeg 支持64路
struct x3_vdec_info_t {
  x3_vdec_chn_info_t m_vdec_chn_info[8]; // 这是个数组
  int m_chn_num;                         // 使能多少个通道
};

// 单个通道的属性配置
struct x3_venc_chn_info_t {
  // 输入数据来源于哪个grp.chn
  int m_vps_grp_id;
  int m_vps_chn_id;
  // vps chn到venc的数据是否bind，如果bind，vps chn输出的数据自动发送给编码器
  int m_is_bind;
  // 编码器属性，内部通过union支持H264、H265、Mjpeg、Jpeg
  int m_venc_chn_id; /* 编码通道 */
  int m_chn_enable;
  int m_enable_transform; // 是否启用改善编码呼吸效应的配置参数
  VENC_CHN_ATTR_S m_chn_attr;

  // 调试字段，把编码后的的h264、h265写到文件中
  int m_is_save_to_file;
  int m_file_name[128];
};

// 定义一组编码器，h264、h265最大32路编码器，jpeg、mjpeg 支持64路
struct x3_venc_info_t {
  x3_venc_chn_info_t m_venc_chn_info[32];
  int m_chn_num; // 使能多少个通道
};

struct x3_venc_en_chns_info_t {
  int m_chn_num;
  VENC_CHN m_enable_chn_idx[VENC_MAX_CHN_NUM];
};

void print_debug_infos(void);

int x3_venc_get_en_chn_info_wrap(x3_venc_info_t *venc_info,
                                 x3_venc_en_chns_info_t *venc_en_chns_info);

int x3_venc_init_wrap(x3_venc_info_t *venc_info);
int x3_venc_uninit_wrap(x3_venc_info_t *venc_info);
int x3_vdec_init_wrap(x3_vdec_chn_info_t *vdec_chn_info);
int x3_vdec_uninit_wrap(x3_vdec_chn_info_t *vdec_chn_info);

#endif // X3_SDK_WRAP_H_
