//
// Created by Wallel on 2022/2/28.
//

#ifndef METAENGINE_RKMEDIARTSPMODULE_H
#define METAENGINE_RKMEDIARTSPMODULE_H

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

#include "module.hpp"
#include "frameMessage.pb.h"

class RKMediaRTSPModule : public Module
{
protected:
    // basic
    int videoHeight, videoWidth;
    bool initSuccess;

    // rga
    RGA_ATTR_S stRgaAttr;

    // venc
    VENC_CHN_ATTR_S venc_chn_attr;

    // buffer
    MB_POOL_PARAM_S stBufferPoolParam;
    MEDIA_BUFFER_POOL mbp;

    // bind and connect
    MPP_CHN_S stSrcChn;
    MPP_CHN_S stDestChn;
    MPP_CHN_S stEncChn;

public:
    RKMediaRTSPModule(Backend *ptr,
                      const std::string &streamName,
                      int width,
                      int height,
                      const std::string &initName,
                      const std::string &initType,
                      const std::vector<std::string> &recv = {},
                      const std::vector<std::string> &send = {},
                      const std::vector<std::string> &pool = {});

    ~RKMediaRTSPModule();

    void
    forward(std::vector<std::tuple<std::string, std::string, queueMessage>> message) override;
};


#endif //METAENGINE_RKMEDIARTSPMODULE_H
