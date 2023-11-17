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