/**
 * @file VideoService.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-24
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "AppComponent.hpp"
#include "StatusDto.hpp"
#include <oatpp/web/protocol/http/Http.hpp>

#ifndef __CRUD_VIDEO_SERVICE_HPP_
#define __CRUD_VIDEO_SERVICE_HPP_
namespace server::face {
class VideoService {
private:
  using Status = oatpp::web::protocol::http::Status;

public:
  oatpp::Object<StatusDto> createStream(oatpp::Int32 const &id,
                                        oatpp::String const &url) {
    auto status = StatusDto::createShared();
    status->status = "OK";
    status->code = 200;
    status->message = "Video was successfully starting";
    return status;
  }

  oatpp::Object<StatusDto> stopVideo(oatpp::Int32 const &id,
                                     oatpp::String const &url) {
    auto status = StatusDto::createShared();
    status->status = "OK";
    status->code = 200;
    status->message = "Video was successfully stopped";
    return status;
  }
};
} // namespace server::face
#endif