/**
 * @file VideoService.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "VideoService.hpp"

namespace server::face {

oatpp::Object<StatusDto>
VideoService::startVideo(oatpp::Object<StreamDto> const streamDto) {
  auto status = StatusDto::createShared();
  // 参数检查
  if (streamDto->name.get() == nullptr || streamDto->libName.get() == nullptr ||
      streamDto->url.get() == nullptr) {
    status->status = "Bad Request";
    status->code = 400;
    status->message = "Parameter error";
    return status;
  }

  // 检查是否是rtsp流
  if (streamDto->url->find("rtsp://") == std::string::npos) {
    // 不是rtsp流，不支持的类型，返回错误
    status->status = "Bad Request";
    status->code = 400;
    status->message = "Not supported video type";
    return status;
  }
  
  auto ret = core::StreamManager::getInstance().registered(
      streamDto->name, streamDto->libName, streamDto->url);
  if (!ret) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "Video startup failed";
  }
  status->status = "OK";
  status->code = 200;
  status->message = "Video was successfully starting";
  return status;
}

oatpp::Object<StatusDto> VideoService::stopVideo(oatpp::String const &name) {

  auto status = StatusDto::createShared();

  if (name.get() == nullptr) {
    status->status = "Bad Request";
    status->code = 400;
    status->message = "Parameter error";
    return status;
  }

  auto ret = core::StreamManager::getInstance().unregistered(name);
  if (!ret) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "Video stop failed";
  }

  status->status = "OK";
  status->code = 200;
  status->message = "Video was successfully stopped";
  return status;
}

} // namespace server::face