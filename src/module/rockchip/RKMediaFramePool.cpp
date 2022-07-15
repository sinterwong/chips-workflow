//
// Created by Wallel on 2022/3/15.
//

#include "RKMediaFramePool.h"

void delRKMediaBuffer(std::vector<std::any> list)
{
    assert(list.size() == 1);
    assert(list[0].has_value());
    assert(list[0].type() == typeid(MEDIA_BUFFER));

    RK_MPI_MB_ReleaseBuffer(std::any_cast<MEDIA_BUFFER>(list[0]));
    list.clear();
}

std::any getPtrRKMediaBuffer(std::vector<std::any> &list, FrameBuf *buf)
{
    assert(list.size() == 1);
    assert(list[0].has_value());
    assert(list[0].type() == typeid(MEDIA_BUFFER));

    return RK_MPI_MB_GetPtr(std::any_cast<MEDIA_BUFFER>(list[0]));
}

std::any getMatRKMediaBuffer(std::vector<std::any> &list, FrameBuf *buf)
{
    assert(list.size() == 1);
    assert(list[0].has_value());
    assert(list[0].type() == typeid(MEDIA_BUFFER));

    void *data = RK_MPI_MB_GetPtr(std::any_cast<MEDIA_BUFFER>(list[0]));
    return cv::Mat(buf->height, buf->width, CV_8UC3, data);
}

std::any getBufferRKMediaBuffer(std::vector<std::any> &list, FrameBuf *buf)
{
    assert(list.size() == 1);
    assert(list[0].has_value());
    assert(list[0].type() == typeid(MEDIA_BUFFER));

    return std::any_cast<MEDIA_BUFFER>(list[0]);
}

FrameBuf makeFrameBuf(MEDIA_BUFFER mb)
{
    FrameBuf message;
    message.write({std::make_any<MEDIA_BUFFER>(mb)},
                  {std::make_pair("void*", getPtrRKMediaBuffer),
                   std::make_pair("Mat", getMatRKMediaBuffer),
                   std::make_pair("MEDIA_BUFFER", getBufferRKMediaBuffer)},
                  delRKMediaBuffer,
                  std::make_tuple(1920, 1080, 3, UINT8));
    return message;
}