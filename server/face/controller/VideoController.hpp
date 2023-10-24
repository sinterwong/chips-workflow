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
  VideoService m_faceService; // Create face service
public:
  static std::shared_ptr<VideoController>
  createShared(OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper)) {
    return std::make_shared<VideoController>(objectMapper);
  }

  ENDPOINT_INFO(createUser) {
    info->summary = "Create new User";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["userId"].description = "User Identifier";
    info->pathParams["url"].description = "Url of user's picture";
  }
  ENDPOINT("GET", "users/create?userId={userId}&url={url}", createUser,
           PATH(Int32, userId), PATH(String, url)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.createUser(userId, url));
  }

  ENDPOINT_INFO(updateUser) {
    info->summary = "Update User by userId and url";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["userId"].description = "User Identifier";
    info->pathParams["url"].description = "Url of user's picture";
  }
  ENDPOINT("GET", "users/update?userId={userId}&url={url}", updateUser,
           PATH(Int32, userId), PATH(String, url)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.updateUser(userId, url));
  }

  ENDPOINT_INFO(deleteUser) {
    info->summary = "Delete User by userId";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["userId"].description = "User Identifier";
  }
  ENDPOINT("DELETE", "users/{userId}", deleteUser, PATH(Int32, userId)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.deleteUser(userId));
  }

  ENDPOINT_INFO(searchUser) {
    info->summary = "Search User by url of user's picture";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["url"].description = "Url of user's picture";
  }
  ENDPOINT("GET", "users/search?url={url}", searchUser, PATH(String, url)) {
    return createDtoResponse(Status::CODE_200, m_faceService.searchUser(url));
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen
} // namespace server::face

#endif