#include <fcntl.h>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include "logger/logger.hpp"
#include "x3_sdk_wrap.hpp"
#include "x3_vio_vdec.hpp"
#include "x3_vio_venc.hpp"
#include "x3_vio_vp.hpp"

void print_file(std::string &&filename) {
  std::string name;
  std::ifstream dataFile("file.txt");
  while (!dataFile.fail() && !dataFile.eof()) {
    dataFile >> name;
    std::cout << name << std::endl;
  }
}

// 打印 vin isp vpu venc等模块的调试信息
void print_debug_infos(void) {
  printf("========================= SIF ==========================\n");
  print_file("/sys/devices/platform/soc/a4001000.sif/cfg_info");
  printf("========================= ISP ==========================\n");
  print_file("/sys/devices/platform/soc/b3000000.isp/isp_status");
  printf("========================= IPU PIPE enable ==============\n");
  print_file("/sys/devices/platform/soc/a4040000.ipu/info/enabled_pipeline");
  printf("========================= IPU PIPE config ==============\n");
  print_file("/sys/devices/platform/soc/a4040000.ipu/info/pipeline0_info");
  print_file("/sys/devices/platform/soc/a4040000.ipu/info/pipeline1_info");
  printf("========================= VENC =========================\n");
  print_file("/sys/kernel/debug/vpu/venc");

  printf("========================= VDEC =========================\n");
  print_file("/sys/kernel/debug/vpu/vdec");

  printf("========================= JENC =========================\n");
  print_file("/sys/kernel/debug/jpu/jenc");

  printf("========================= IAR ==========================\n");
  print_file("/sys/kernel/debug/iar");

  printf("========================= ION ==========================\n");
  print_file("/sys/kernel/debug/ion/heaps/carveout");
  print_file("/sys/kernel/debug/ion/heaps/ion_cma");

  printf("========================= END ===========================\n");
}

int x3_venc_get_en_chn_info_wrap(x3_venc_info_t *venc_info,
                                 x3_venc_en_chns_info_t *venc_en_chns_info) {
  int i = 0;
  for (i = 0; i < venc_info->m_chn_num; i++) {
    if (venc_info->m_venc_chn_info[i].m_chn_enable) {
      venc_en_chns_info->m_chn_num++;
      venc_en_chns_info->m_enable_chn_idx[i] =
          venc_info->m_venc_chn_info[i].m_venc_chn_id;
    }
  }
  return 0;
}

int x3_venc_init_wrap(x3_venc_info_t *venc_info) {
  int ret = 0;
  int i = 0;
  for (i = 0; i < venc_info->m_chn_num; i++) {
    ret = x3_venc_init(venc_info->m_venc_chn_info[i].m_venc_chn_id,
                       &venc_info->m_venc_chn_info[i].m_chn_attr);
    if (ret) {
      printf("x3_venc_init failed, %d\n", ret);
      return -1;
    }
  }
  FLOWENGINE_LOGGER_INFO("ok!");
  return 0;
}

int x3_venc_uninit_wrap(x3_venc_info_t *venc_info) {
  int ret = 0;
  int i = 0;
  for (i = 0; i < venc_info->m_chn_num; i++) {
    ret = x3_venc_deinit(venc_info->m_venc_chn_info[i].m_venc_chn_id);
    if (ret) {
      printf("x3_venc_deinit chn%d failed, %d\n",
             venc_info->m_venc_chn_info[i].m_venc_chn_id, ret);
      return -1;
    }
  }
  return 0;
}

int x3_vdec_init_wrap(x3_vdec_chn_info_t *vdec_chn_info) {
  int ret = 0;
  // 创建内存buff
  x3_vp_alloc(&vdec_chn_info->vp_param);
  // 初始化解码器
  ret = x3_vdec_init(vdec_chn_info->m_vdec_chn_id, &vdec_chn_info->m_chn_attr);
  if (ret) {
    FLOWENGINE_LOGGER_ERROR("x3_vdec_init failed, {}", ret);
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("start vdec chn{} ok!", vdec_chn_info->m_vdec_chn_id);
  return 0;
}

int x3_vdec_uninit_wrap(x3_vdec_chn_info_t *vdec_chn_info) {
  int ret = 0;
  ret = x3_vdec_deinit(vdec_chn_info->m_vdec_chn_id);
  if (ret) {
    printf("x3_vdec_deinit failed, %d\n", ret);
    return -1;
  }
  // 释放内存buff
  x3_vp_free(&vdec_chn_info->vp_param);
  return 0;
}
