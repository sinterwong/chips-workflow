/**
 * @file FaceController.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "FaceService.hpp"
#include "FacelibDto.hpp"
#include "StatusDto.hpp"
#include <memory>
#include <oatpp/core/macro/codegen.hpp>
#include <oatpp/parser/json/mapping/ObjectMapper.hpp>
#include <oatpp/web/server/api/ApiController.hpp>

#ifndef __CRUD_FACE_CONTROLLER_HPP_
#define __CRUD_FACE_CONTROLLER_HPP_
namespace server::face {
#include OATPP_CODEGEN_BEGIN(ApiController) //<- Begin Codegen
class FaceController : public oatpp::web::server::api::ApiController {
public:
  FaceController(OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper))
      : oatpp::web::server::api::ApiController(objectMapper) {}

private:
  FaceService m_faceService; // Create face service
public:
  static std::shared_ptr<FaceController>
  createShared(OATPP_COMPONENT(std::shared_ptr<ObjectMapper>, objectMapper)) {
    return std::make_shared<FaceController>(objectMapper);
  }

  ENDPOINT_INFO(createOneUser) {
    info->summary = "Create new User";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["userId"].description = "User Identifier";
    info->pathParams["url"].description = "Url of user's picture";
  }
  ENDPOINT("GET", "facelib/v0/createOne", createOneUser, QUERY(Int32, userId),
           QUERY(String, url)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.createUser(userId, url));
  }

  ENDPOINT_INFO(updateOneUser) {
    info->summary = "Update User by userId and url";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["userId"].description = "User Identifier";
    info->pathParams["url"].description = "Url of user's picture";
  }
  ENDPOINT("GET", "facelib/v0/update", updateOneUser, QUERY(Int32, userId),
           QUERY(String, url)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.updateUser(userId, url));
  }

  ENDPOINT_INFO(deleteOneUser) {
    info->summary = "Delete User by userId";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["userId"].description = "User Identifier";
  }
  ENDPOINT("DELETE", "facelib/v0/{userId}", deleteOneUser, PATH(Int32, userId)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.deleteUser(userId));
  }

  // 以图搜人
  ENDPOINT_INFO(searchUser) {
    info->summary = "Search User by url of user's picture";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["url"].description = "Url of user's picture";
  }
  ENDPOINT("GET", "facelib/v0/search", searchUser, QUERY(String, url)) {
    return createDtoResponse(Status::CODE_200, m_faceService.searchUser(url));
  }

  // 批量新增
  ENDPOINT_INFO(createBatchUsers) {
    info->summary = "Create batch users";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("POST", "facelib/v0/createBatch", createBatchUsers,
           BODY_DTO(Object<FacelibDto>, users)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.createBatch(users));
  }

  // 批量更新
  ENDPOINT_INFO(updateBatchUsers) {
    info->summary = "Update batch users";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("POST", "facelib/v0/updateBatch", updateBatchUsers,
           BODY_DTO(Object<FacelibDto>, users)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.updateBatch(users));
  }

  // 批量删除
  ENDPOINT_INFO(deleteBatchUsers) {
    info->summary = "Delete batch users";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("POST", "facelib/v0/deleteBatch", deleteBatchUsers,
           BODY_DTO(Object<FacelibDto>, users)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.deleteBatch(users));
  }

  // 两图比对
  ENDPOINT_INFO(compareTwoPictures) {
    info->summary = "Compare two pictures";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("GET", "facelib/v0/compare", compareTwoPictures,
           QUERY(String, url1), QUERY(String, url2)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.compareTwoPictures(url1, url2));
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen
} // namespace server::face
#endif