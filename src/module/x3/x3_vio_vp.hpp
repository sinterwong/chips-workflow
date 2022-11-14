/***************************************************************************
 * COPYRIGHT NOTICE
 * Copyright 2020 Horizon Robotics, Inc.
 * All rights reserved.
 ***************************************************************************/
#ifndef X3_VIO_VP_H_
#define X3_VIO_VP_H_
#include "x3_sdk_wrap.hpp"

int x3_vp_init();
int x3_vp_alloc(vp_param_t *param);
int x3_vp_free(vp_param_t *param);
int x3_vp_deinit();

#endif // X3_VIO_VP_H_
