/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#include "logger/logger.hpp"
#include "stdint.h"
#include <cstring>
#include <stdio.h>
#include <unistd.h>

#include "vio/hb_sys.h"
#include "vio/hb_vp_api.h"

#include "x3_vio_vp.hpp"

int x3_vp_init() {
  VP_CONFIG_S struVpConf;
  memset(&struVpConf, 0x00, sizeof(VP_CONFIG_S));
  struVpConf.u32MaxPoolCnt = 32; // 整个系统中可以容纳缓冲池的个数
  HB_VP_SetConfig(&struVpConf);

  int ret = HB_VP_Init();
  if (!ret) {
    printf("hb_vp_init success\n");
  } else {
    printf("hb_vp_init failed, ret: %d\n", ret);
  }
  return ret;
}

int x3_vp_alloc(vp_param_t *param) {
  int i = 0, ret = 0;
  vp_param_t *vp_param = (vp_param_t *)param;

  for (i = 0; i < vp_param->mmz_cnt; i++) {

    ret = HB_SYS_Alloc(&vp_param->mmz_paddr[i],
                       (void **)&vp_param->mmz_vaddr[i], vp_param->mmz_size);
    if (!ret) {
      FLOWENGINE_LOGGER_INFO("mmzAlloc paddr = 0x{}, vaddr = 0x{}, {}/{} , {}",
                             vp_param->mmz_paddr[i], vp_param->mmz_vaddr[i], i,
                             vp_param->mmz_cnt, vp_param->mmz_size);
    } else {
      FLOWENGINE_LOGGER_ERROR("hb_vp_alloc failed, ret: {}", ret);
      return -1;
    }
  }
  return 0;
}

int x3_vp_free(vp_param_t *param) {
  int i = 0, ret = 0;
  vp_param_t *vp_param = (vp_param_t *)param;
  for (i = 0; i < vp_param->mmz_cnt; i++) {
    ret = HB_SYS_Free(vp_param->mmz_paddr[i], vp_param->mmz_vaddr[i]);
    if (ret == 0) {
      FLOWENGINE_LOGGER_INFO("mmzFree paddr = 0x{}, vaddr = 0x{} i = {}",
                             vp_param->mmz_paddr[i], vp_param->mmz_vaddr[i], i);
    }
  }
  return 0;
}

int x3_vp_deinit() {
  int ret = HB_VP_Exit();
  if (!ret) {
    printf("hb_vp_deinit success\n");
  } else {
    printf("hb_vp_deinit failed, ret: %d\n", ret);
  }
  return ret;
}
