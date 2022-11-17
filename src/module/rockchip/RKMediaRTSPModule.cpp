#include "RKMediaRTSPModule.h"

// rtsp
rtsp_demo_handle g_rtsplive;
rtsp_session_handle g_rtsp_session;

void video_packet_cb(MEDIA_BUFFER mb)
{
    static RK_S32 packet_cnt = 0;

    printf("#Get packet-%d, size %zu\n", packet_cnt, RK_MPI_MB_GetSize(mb));

    if (g_rtsplive && g_rtsp_session)
    {
        rtsp_tx_video(g_rtsp_session, (u_char *) RK_MPI_MB_GetPtr(mb),
                      RK_MPI_MB_GetSize(mb),
                      RK_MPI_MB_GetTimestamp(mb));
        rtsp_do_event(g_rtsplive);
    }

    RK_MPI_MB_ReleaseBuffer(mb);
    packet_cnt++;
}

RKMediaRTSPModule::RKMediaRTSPModule(Backend *ptr,
                                     const std::string &streamName,
                                     int width,
                                     int height,
                                     const std::string &initName,
                                     const std::string &initType,
                                     
                                     )
        : Module(ptr, initName, initType)
{
    bool ret;
    initSuccess = true;
    videoHeight = height;
    videoWidth = width;

    // RGA init
    stRgaAttr.bEnBufPool = RK_TRUE;
    stRgaAttr.u16BufPoolCnt = 2;
    stRgaAttr.bEnBufPool = RK_TRUE;
    stRgaAttr.u16BufPoolCnt = 2;
    stRgaAttr.u16Rotaion = 0;
    stRgaAttr.stImgIn.u32X = 0;
    stRgaAttr.stImgIn.u32Y = 0;
    stRgaAttr.stImgIn.imgType = IMAGE_TYPE_BGR888;
    stRgaAttr.stImgIn.u32Width = videoWidth;
    stRgaAttr.stImgIn.u32Height = videoHeight;
    stRgaAttr.stImgIn.u32HorStride = videoWidth;
    stRgaAttr.stImgIn.u32VirStride = videoHeight;
    stRgaAttr.stImgOut.u32X = 0;
    stRgaAttr.stImgOut.u32Y = 0;
    stRgaAttr.stImgOut.imgType = IMAGE_TYPE_NV12;
    stRgaAttr.stImgOut.u32Width = videoWidth;
    stRgaAttr.stImgOut.u32Height = videoHeight;
    stRgaAttr.stImgOut.u32HorStride = videoWidth;
    stRgaAttr.stImgOut.u32VirStride = videoHeight;
    ret = RK_MPI_RGA_CreateChn(1, &stRgaAttr);
    if (ret)
    {
        printf("Create RTSP failed! ret=%d\n", ret);
        initSuccess = false;
    }

    //VENC init
    memset(&venc_chn_attr, 0, sizeof(venc_chn_attr));
    venc_chn_attr.stVencAttr.enType = RK_CODEC_TYPE_H264;
    venc_chn_attr.stRcAttr.enRcMode = VENC_RC_MODE_H264CBR;
    venc_chn_attr.stRcAttr.stH264Cbr.u32Gop = 30;
    venc_chn_attr.stRcAttr.stH264Cbr.u32BitRate = videoWidth * videoHeight;
    // frame rate: in 30/1, out 30/1.
    venc_chn_attr.stRcAttr.stH264Cbr.fr32DstFrameRateDen = 1;
    venc_chn_attr.stRcAttr.stH264Cbr.fr32DstFrameRateNum = 30;
    venc_chn_attr.stRcAttr.stH264Cbr.u32SrcFrameRateDen = 1;
    venc_chn_attr.stRcAttr.stH264Cbr.u32SrcFrameRateNum = 30;
    venc_chn_attr.stVencAttr.imageType = IMAGE_TYPE_NV12;
    venc_chn_attr.stVencAttr.u32PicWidth = videoWidth;
    venc_chn_attr.stVencAttr.u32PicHeight = videoHeight;
    venc_chn_attr.stVencAttr.u32VirWidth = videoWidth;
    venc_chn_attr.stVencAttr.u32VirHeight = videoHeight;
    venc_chn_attr.stVencAttr.u32Profile = 77;
    ret = RK_MPI_VENC_CreateChn(0, &venc_chn_attr);
    if (ret)
    {
        printf("ERROR: create VENC[0] error! ret=%d\n", ret);
        initSuccess = false;
    }

    // init buffer pool
    stBufferPoolParam.u32Cnt = 3;
    stBufferPoolParam.u32Size =
            0; // Automatic calculation using imgInfo internally
    stBufferPoolParam.enMediaType = MB_TYPE_VIDEO;
    stBufferPoolParam.bHardWare = RK_TRUE;
    stBufferPoolParam.u16Flag = MB_FLAG_NOCACHED;
    stBufferPoolParam.stImageInfo.enImgType = IMAGE_TYPE_BGR888;
    stBufferPoolParam.stImageInfo.u32Width = videoWidth;
    stBufferPoolParam.stImageInfo.u32Height = videoHeight;
    stBufferPoolParam.stImageInfo.u32HorStride = videoWidth;
    stBufferPoolParam.stImageInfo.u32VerStride = videoHeight;

    mbp = RK_MPI_MB_POOL_Create(&stBufferPoolParam);
    if (!mbp)
    {
        printf("Create buffer pool for vo failed!\n");
        initSuccess = false;
    }

    // init rtsp
    g_rtsplive = create_rtsp_demo(554);
    if (not streamName.empty())
        g_rtsp_session = rtsp_new_session(g_rtsplive, streamName.c_str());
    else
        g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/main_stream");
    rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H264, NULL, 0);
    rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

    // bind RGA and VENC
    stSrcChn.enModId = RK_ID_RGA;
    stSrcChn.s32DevId = 0;
    stSrcChn.s32ChnId = 1;
    stDestChn.enModId = RK_ID_VENC;
    stDestChn.s32DevId = 0;
    stDestChn.s32ChnId = 0;
    ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);

    // Register Out Function
    stEncChn.enModId = RK_ID_VENC;
    stEncChn.s32DevId = 0;
    stEncChn.s32ChnId = 0;
    ret = RK_MPI_SYS_RegisterOutCb(&stEncChn, video_packet_cb);

    printf("RKMediaRTSPModule Init Finish!\n");
}

RKMediaRTSPModule::~RKMediaRTSPModule()
{
    bool ret;

    if (g_rtsplive)
        rtsp_del_demo(g_rtsplive);

    ret = RK_MPI_SYS_UnBind(&stSrcChn, &stDestChn);
    ret = RK_MPI_VENC_DestroyChn(0);
    ret = RK_MPI_RGA_DestroyChn(1);
    ret = RK_MPI_MB_POOL_Destroy(mbp);
}

void RKMediaRTSPModule::forward(
        std::vector<forwardMessage> message)
{
    for (auto&[send, type, buf]: message)
    {
        assert(type == "stream");
        auto frameBufMessage = backendPtr->pool->read(buf.key);

        assert(height == videoHeight);
        assert(width == videoWidth);

        auto mb = std::any_cast<MEDIA_BUFFER>(frameBufMessage.read("MEDIA_BUFFER"));
        RK_MPI_SYS_SendMediaBuffer(RK_ID_RGA, 1, mb);
    }
}


