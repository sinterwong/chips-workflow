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
#include "StreamDto.hpp"
#include "streamManager.hpp"
#include <oatpp/web/protocol/http/Http.hpp>

#ifndef __CRUD_VIDEO_SERVICE_HPP_
#define __CRUD_VIDEO_SERVICE_HPP_
namespace server::face {
class VideoService {
private:
  using Status = oatpp::web::protocol::http::Status;

public:
  // 启动视频流
  oatpp::Object<StatusDto> startVideo(oatpp::Object<StreamDto> const streamDto);

  // 停止视频流
  oatpp::Object<StatusDto> stopVideo(oatpp::String const &name);
};
} // namespace server::face
#endif