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
    FLOWENGINE_LOGGER_INFO("hb_vp_init success");
  } else {
    FLOWENGINE_LOGGER_ERROR("hb_vp_init failed, ret: {}", ret);
  }
  return ret;
}

int x3_vp_deinit() {
  int ret = HB_VP_Exit();
  if (!ret) {
    FLOWENGINE_LOGGER_INFO("hb_vp_deinit success");
  } else {
    FLOWENGINE_LOGGER_ERROR("hb_vp_deinit failed, ret: {}", ret);
  }
  return ret;
}
