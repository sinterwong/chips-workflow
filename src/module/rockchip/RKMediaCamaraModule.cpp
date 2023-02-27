#include "RKMediaCameraModule.h"


RKMediaCameraModule::RKMediaCameraModule(backend_ptr ptr,
                                         const std::string &iqfile,
                                         std::string const &name,
                                         std::string const &type,
                                         
                                             )
        : Module(ptr, name, type)
{
    bool ret;
    initSuccess = true;

    if (not iqfile.empty())
    {
        pIqfilesPath = iqfile;
    }

    printf("#####Device: %s\n", pDeviceName.c_str());
    printf("#####Resolution: %dx%d\n", u32Width, u32Height);
    printf("#####Frame Count to save: %d\n", frameCnt);
    printf("#####Output Path: %s\n", pOutPath.c_str());
    printf("#CameraIdx: %d\n\n", s32CamId);

    if (not pIqfilesPath.empty())
    {
#ifdef RKAIQ
        printf("#####Aiq xml dirpath: %s\n\n", pIqfilesPath.c_str());
        printf("#bMultictx: %d\n\n", bMultictx);
        rk_aiq_working_mode_t hdr_mode = RK_AIQ_WORKING_MODE_NORMAL;
        int fps = 30;
        SAMPLE_COMM_ISP_Init(s32CamId, hdr_mode, bMultictx,
                             pIqfilesPath.c_str());
        SAMPLE_COMM_ISP_Run(s32CamId);
        SAMPLE_COMM_ISP_SetFrameRate(s32CamId, fps);
#endif
    }

    RK_MPI_SYS_Init();
    vi_chn_attr.pcVideoNode = pDeviceName.c_str();
    vi_chn_attr.u32BufCnt = 7;
    vi_chn_attr.u32Width = u32Width;
    vi_chn_attr.u32Height = u32Height;
    vi_chn_attr.enPixFmt = IMAGE_TYPE_NV12;
    vi_chn_attr.enWorkMode = VI_WORK_MODE_NORMAL;
    vi_chn_attr.enBufType = VI_CHN_BUF_TYPE_MMAP;
    ret = RK_MPI_VI_SetChnAttr(s32CamId, 0, &vi_chn_attr);
    ret |= RK_MPI_VI_EnableChn(s32CamId, 0);
    if (ret)
    {
        printf("Create VI[0] failed! ret=%d\n", ret);
        initSuccess = false;
    }

    stRgaAttr.bEnBufPool = RK_TRUE;
    stRgaAttr.u16BufPoolCnt = 7;
    stRgaAttr.u16Rotaion = 0;
    stRgaAttr.stImgIn.u32X = 0;
    stRgaAttr.stImgIn.u32Y = 0;
    stRgaAttr.stImgIn.imgType = IMAGE_TYPE_NV12;
    stRgaAttr.stImgIn.u32Width = u32Width;
    stRgaAttr.stImgIn.u32Height = u32Height;
    stRgaAttr.stImgIn.u32HorStride = u32Width;
    stRgaAttr.stImgIn.u32VirStride = u32Height;
    stRgaAttr.stImgOut.u32X = 0;
    stRgaAttr.stImgOut.u32Y = 0;
    stRgaAttr.stImgOut.imgType = IMAGE_TYPE_BGR888;
    stRgaAttr.stImgOut.u32Width = u32Width;
    stRgaAttr.stImgOut.u32Height = u32Height;
    stRgaAttr.stImgOut.u32HorStride = u32Width;
    stRgaAttr.stImgOut.u32VirStride = u32Height;
    ret = RK_MPI_RGA_CreateChn(0, &stRgaAttr);
    if (ret)
    {
        printf("Create RGA[0] failed! ret=%d\n", ret);
        initSuccess = false;
    }


    stSrcChn.enModId = RK_ID_VI;
    stSrcChn.s32DevId = s32CamId;
    stSrcChn.s32ChnId = 0;
    stDestChn.enModId = RK_ID_RGA;
    stDestChn.s32DevId = 0;
    stDestChn.s32ChnId = 0;
    ret = RK_MPI_SYS_Bind(&stSrcChn, &stDestChn);

    if (ret)
    {
        printf("Bind VI to RGA[0] failed! ret=%d\n", ret);
        initSuccess = false;
    }

    printf("RKMediaCamaraModule Init Finish!\n");
}

void RKMediaCameraModule::forward(
        std::vector<forwardMessage> &message)
{
    ptr->pool->checkSize();
    mb = RK_MPI_SYS_GetMediaBuffer(RK_ID_RGA, 0, -1);
    if (!mb)
    {
        printf("RK_MPI_SYS_GetMediaBuffer get null buffer!\n");
    } else
    {
        MB_IMAGE_INFO_S stImageInfo;
        int ret = RK_MPI_MB_GetImageInfo(mb, &stImageInfo);
        if (ret)
            printf("Warn: Get image info failed! ret = %d\n", ret);
        FrameBuf frameBufMessage = makeFrameBuf(mb);
        int returnKey = ptr->pool->write(frameBufMessage);

        queueMessage sendMessage;
        sendMessage.width = stImageInfo.u32Width;
        sendMessage.height = stImageInfo.u32Height;
        sendMessage.type = "BGA888";
        sendMessage.key = returnKey;

        autoSend(sendMessage);
        printf("moving\n");
    }
}

