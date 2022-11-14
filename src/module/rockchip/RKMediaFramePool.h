#ifndef METAENGINE_RKMEDIAFRAMEPOOL_H
#define METAENGINE_RKMEDIAFRAMEPOOL_H

#include "routeFramePool.h"
#include "rkmedia_api.h"
#include "rkmedia_venc.h"

#include <any>
#include <vector>
#include <opencv2/opencv.hpp>

void delRKMediaBuffer(std::vector<std::any> list);
std::any getPtrRKMediaBuffer(std::vector<std::any> &list, FrameBuf* buf);
std::any getMatRKMediaBuffer(std::vector<std::any> &list, FrameBuf* buf);
std::any getBufferRKMediaBuffer(std::vector<std::any> &list, FrameBuf* buf);

FrameBuf makeFrameBuf(MEDIA_BUFFER mb);

#endif //METAENGINE_RKMEDIAFRAMEPOOL_H
