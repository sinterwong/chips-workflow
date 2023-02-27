#ifndef METAENGINE_RKMEDIACAMERAMODULE_H
#define METAENGINE_RKMEDIACAMERAMODULE_H

#include <easymedia/rkmedia_api.h>

#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <iostream>

#include "sample_common.h"
#include "rtsp_demo.h"
#include "rkmedia_api.h"
#include "rkmedia_venc.h"

#include "RockchipRga.h"
#include "RgaUtils.h"
#include "im2d.h"

#include "module.hpp"
#include "basicMessage.pb.h"
#include "frameMessage.pb.h"
#include "RKMediaFramePool.h"

class RKMediaCameraModule : public Module
{
protected:
    bool initSuccess = true;
    VO_CHN_ATTR_S stVoAttr;
    RK_U32 u32Width = 1920;
    RK_U32 u32Height = 1080;
    int frameCnt = -1;
    std::string pDeviceName = "rkispp_scale0";
    std::string pOutPath = "";
    std::string pIqfilesPath = "/etc/iqfiles/";
    RK_S32 s32CamId = 0;
#ifdef RKAIQ
    RK_BOOL bMultictx = RK_FALSE;
#endif
    VI_CHN_ATTR_S vi_chn_attr;
    RGA_ATTR_S stRgaAttr;
    MPP_CHN_S stDestChn;
    MPP_CHN_S stSrcChn;
    MEDIA_BUFFER mb = NULL;
    tutorial::FrameMessage buf;
public:
    RKMediaCameraModule(backend_ptr ptr,
                       const std::string &iqfile,
                       std::string const &name,
                       std::string const &type,
                       
                               );

    void
    forward(std::vector<forwardMessage> &message) override;
};


#endif //METAENGINE_RKMEDIACAMERAMODULE_H
