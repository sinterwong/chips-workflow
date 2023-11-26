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
#include "FaceDto.hpp"
#include "FaceService.hpp"
#include "FacelibDto.hpp"
#include "ImageDto.hpp"
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

  // RESTful API，GET 请求通常用于获取资源，POST 通常用于创建资源。
  // 幂等性是指一次和多次请求某一个资源应该具有同样的副作用。GET、DELETE和PUT应该是幂等的，而POST不是。
  ENDPOINT_INFO(createOneUser) {
    info->summary = "Create new User";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("POST", "face/v0/facelib/createOne", createOneUser,
           BODY_DTO(Object<FaceDto>, user)) {
    return createDtoResponse(Status::CODE_200, m_faceService.createUser(user));
  }

  ENDPOINT_INFO(updateOneUser) {
    info->summary = "Update User by userId and url";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("POST", "face/v0/facelib/updateOne", updateOneUser,
           BODY_DTO(Object<FaceDto>, user)) {
    return createDtoResponse(Status::CODE_200, m_faceService.updateUser(user));
  }

  ENDPOINT_INFO(deleteOneUser) {
    info->summary = "Delete User by userId";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["userId"].description = "User Identifier";
  }
  ENDPOINT("DELETE", "face/v0/facelib/{userId}", deleteOneUser,
           PATH(String, userId)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.deleteUser(userId));
  }

  // 以图搜人 GET
  ENDPOINT_INFO(searchUser) {
    info->summary = "Search User by url of user's picture";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");

    info->pathParams["libName"].description = "Name of face library";
    info->pathParams["url"].description = "Url of user's picture";
  }
  ENDPOINT("GET", "face/v0/facelib/search", searchUser, QUERY(String, libName),
           QUERY(String, url)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.searchUser(libName, url));
  }

  // 以图搜人 POST
  ENDPOINT_INFO(searchUserPost) {
    info->summary = "Search User by url of user's picture";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("POST", "face/v0/facelib/search", searchUserPost,
           BODY_DTO(Object<ImageDto>, image)) {
    return createDtoResponse(Status::CODE_200, m_faceService.searchUser(image));
  }

  // 两图比对
  ENDPOINT_INFO(compareTwoPictures) {
    info->summary = "Compare two pictures";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("GET", "face/v0/facelib/compare", compareTwoPictures,
           QUERY(String, url1), QUERY(String, url2)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.compareTwoPictures(url1, url2));
  }

  // 两图比对
  ENDPOINT_INFO(compareTwoPicturesPost) {
    info->summary = "Compare two pictures";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }
  ENDPOINT("POST", "face/v0/facelib/compare", compareTwoPicturesPost,
           BODY_DTO(oatpp::Vector<Object<ImageDto>>, images)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.compareTwoPictures(images));
  }

  // 批量新增
  ENDPOINT_INFO(createBatchUsers) {
    info->summary = "Create batch users";
    info->addResponse<Object<StatusDto>>(Status::CODE_200, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_404, "application/json");
    info->addResponse<Object<StatusDto>>(Status::CODE_500, "application/json");
  }

  ENDPOINT("POST", "face/v0/facelib/createBatch", createBatchUsers,
           BODY_DTO(oatpp::Vector<Object<FaceDto>>, users)) {
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
  ENDPOINT("POST", "face/v0/facelib/updateBatch", updateBatchUsers,
           BODY_DTO(oatpp::Vector<Object<FaceDto>>, users)) {
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
  ENDPOINT("POST", "face/v0/facelib/deleteBatch", deleteBatchUsers,
           BODY_DTO(oatpp::Vector<Object<FaceDto>>, users)) {
    return createDtoResponse(Status::CODE_200,
                             m_faceService.deleteBatch(users));
  }
};

#include OATPP_CODEGEN_END(ApiController) //<- End Codegen
} // namespace server::face
#endif