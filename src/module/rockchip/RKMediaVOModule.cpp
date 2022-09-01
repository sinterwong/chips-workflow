//
// Created by Wallel on 2022/2/28.
//

#include "RKMediaVOModule.h"

RKMediaVOModule::RKMediaVOModule(Backend *ptr,
                                 const std::string &streamName,
                                 std::tuple<int, int> videoSize,
                                 std::tuple<int, int> screenSize,
                                 const std::string &initName,
                                 const std::string &initType,
                                 const std::vector<std::string> &recv,
                                 const std::vector<std::string> &send      )
        : Module(ptr, initName, initType, recv, send)
{
    bool ret;
    initSuccess = true;
    std::tie(videoWidth, videoHeight) = videoSize;
    std::tie(screenWidth, screenHeight) = screenSize;

    //VO init
    memset(&stVoAttr, 0, sizeof(stVoAttr));
    stVoAttr.pcDevNode = "/dev/dri/card0";
    stVoAttr.u32Width = screenHeight;
    stVoAttr.u32Height = screenWidth;
    stVoAttr.u16Fps = 0;
    stVoAttr.u16Zpos = 0;
    stVoAttr.stImgRect.s32X = 0;
    stVoAttr.stImgRect.s32Y = 0;
    stVoAttr.stImgRect.u32Width = screenHeight;
    stVoAttr.stImgRect.u32Height = screenWidth;
    stVoAttr.stDispRect.s32X = 0;
    stVoAttr.stDispRect.s32Y = 0;
    stVoAttr.stDispRect.u32Width = screenHeight;
    stVoAttr.stDispRect.u32Height = screenWidth;
    stVoAttr.emPlaneType = VO_PLANE_PRIMARY;
    stVoAttr.enImgType = IMAGE_TYPE_BGR888;
    fltImgRatio = 3.0; // for RGB888
    ret = RK_MPI_VO_CreateChn(0, &stVoAttr);
    if (ret)
    {
        printf("ERROR: create VO[0] error! ret=%d\n", ret);
        initSuccess = false;
    }

    printf("RKMediaRTSPModule Init Finish!\n");
}

RKMediaVOModule::~RKMediaVOModule()
{
    bool ret;
    ret = RK_MPI_VO_DestroyChn(0);
}

void RKMediaVOModule::forward(
        std::vector<std::tuple<std::string, std::string, queueMessage>> message)
{
    bool ret;
    for (auto&[send, type, buf]: message)
    {
        assert(type == "stream");

        auto frameBufMessage = backendPtr->pool->read(buf.key);

        assert(height == videoHeight);
        assert(width == videoWidth);

        MB_IMAGE_INFO_S srcImageInfo = {
                static_cast<RK_U32>(videoWidth),
                static_cast<RK_U32>(videoHeight),
                static_cast<RK_U32>(videoWidth),
                static_cast<RK_U32>(videoHeight), IMAGE_TYPE_BGR888};

       MEDIA_BUFFER src_mb = std::any_cast<MEDIA_BUFFER>(frameBufMessage.read("MEDIA_BUFFER"));
        bool createSrcMb = true;

//        MEDIA_BUFFER src_mb = RK_MPI_MB_CreateImageBuffer(&srcImageInfo,
//                                             RK_TRUE,
//                                             0);
        if (!src_mb)
        {
            printf("WARN: BufferPool get null buffer...\n");
        }

//        memcpy(RK_MPI_MB_GetPtr(src_mb),
//               frameBufMessage.read(),
//               sizeof(unsigned char) * frameBufMessage.size());
//        RK_MPI_MB_SetSize(src_mb,
//                          sizeof(unsigned char) * frameBufMessage.size());


        MB_IMAGE_INFO_S dstImageInfo = {
                static_cast<RK_U32>(screenWidth),
                static_cast<RK_U32>(screenHeight),
                static_cast<RK_U32>(screenWidth),
                static_cast<RK_U32>(screenHeight), IMAGE_TYPE_BGR888};
        MEDIA_BUFFER dst_mb = RK_MPI_MB_CreateImageBuffer(&dstImageInfo,
                                                          RK_TRUE,
                                                          0);

        rga_buffer_t src = wrapbuffer_fd(RK_MPI_MB_GetFD(src_mb), videoWidth,
                                         videoHeight, RK_FORMAT_BGR_888);
        rga_buffer_t dst = wrapbuffer_fd(RK_MPI_MB_GetFD(dst_mb), screenWidth,
                                         screenHeight, RK_FORMAT_BGR_888);

        ret = imresize(src, dst);

        MB_IMAGE_INFO_S rotationImageInfo = {
                static_cast<RK_U32>(screenHeight),
                static_cast<RK_U32>(screenWidth),
                static_cast<RK_U32>(screenHeight),
                static_cast<RK_U32>(screenWidth), IMAGE_TYPE_BGR888};
        MEDIA_BUFFER rotation_mb = RK_MPI_MB_CreateImageBuffer(
                &rotationImageInfo,
                RK_TRUE,
                0);

        rga_buffer_t rotation_dst = wrapbuffer_fd(RK_MPI_MB_GetFD(rotation_mb),
                                                  screenHeight,
                                                  screenWidth,
                                                  RK_FORMAT_BGR_888);
        ret = imrotate(dst, rotation_dst, IM_HAL_TRANSFORM_ROT_90);

        MB_IMAGE_INFO_S RGBImageInfo = {
                static_cast<RK_U32>(screenHeight),
                static_cast<RK_U32>(screenWidth),
                static_cast<RK_U32>(screenHeight),
                static_cast<RK_U32>(screenWidth), IMAGE_TYPE_RGB888};
        MEDIA_BUFFER rgb_mb = RK_MPI_MB_CreateImageBuffer(
                &RGBImageInfo,
                RK_TRUE,
                0);
        rga_buffer_t rgb_dst = wrapbuffer_fd(RK_MPI_MB_GetFD(rgb_mb),
                                             screenHeight,
                                             screenWidth,
                                             RK_FORMAT_BGR_888);
        ret = imcvtcolor(rotation_dst, rgb_dst, RK_FORMAT_BGR_888,
                         RK_FORMAT_RGB_888);

        RK_MPI_SYS_SendMediaBuffer(RK_ID_VO, 0, rgb_mb);

//        RK_MPI_MB_ReleaseBuffer(src_mb);
        RK_MPI_MB_ReleaseBuffer(dst_mb);
        RK_MPI_MB_ReleaseBuffer(rotation_mb);
        RK_MPI_MB_ReleaseBuffer(rgb_mb);
        printf("moving!\n");
    }
}



