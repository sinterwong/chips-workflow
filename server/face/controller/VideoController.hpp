/**
 * @file VideoController.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "StatusDto.hpp"
#include "VideoService.hpp"
#include <memory>
#include <oatpp/core/macro/codegen.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/api/ApiController.hpp>

#ifndef __CRUD_FACE_CONTROLLER_HPP_
#define __CRUD_FACE_CONTROLLER_HPP_

namespace server::face {
#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen
class VideoController : public oatpp::web::server::api::ApiController {
public:
  VideoController(OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper))
      : oatpp::web::server::api::ApiController(objectMapper) {}

private:
  VideoService m_videoService; // Create face service
public:
  static std::shared_ptr<VideoController>
  createShared(OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper)) {
    return std::make_shared<VideoController>(objectMapper);
  }

  ENDPOINT_INFO(startVideo) {
    info->summary = "Create a new stream";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["name"].description = "Video stream's Identifier";
    info->pathParams["url"].description = "Url of the stream";
  }
  ENDPOINT("GET", "users/startVideo?name={name}&url={url}", startVideo,
           PATH(String, name), PATH(String, url)) {
    return createDtoResponse(Status::CODE_200,
                             m_videoService.startVideo(name, url));
  }

  ENDPOINT_INFO(stopVideo) {
    info->summary = "Stop the stream by a name";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["name"].description = "Video stream's Identifier";
  }
  ENDPOINT("GET", "users/stopVideo?name={name}", stopVideo,
           PATH(String, name)) {
    return createDtoResponse(Status::CODE_200, m_videoService.stopVideo(name));
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen
} // namespace server::face

#endif
