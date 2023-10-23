/**
 * @file faceService.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "AppComponent.hpp"
#include "StatusDto.hpp"
#include <oatpp/web/protocol/http/Http.hpp>

#ifndef __CRUD_FACE_SERVICE_HPP_
#define __CRUD_FACE_SERVICE_HPP_

class FaceService {
private:
  using Status = oatpp::web::protocol::http::Status;

public:
  oatpp::Object<StatusDto> createUser(oatpp::Int32 const &id,
                                      oatpp::String const &url) {
    auto status = StatusDto::createShared();
    status->status = "OK";
    status->code = 200;
    status->message = "User was successfully created";
    return status;
  }
  oatpp::Object<StatusDto> updateUser(oatpp::Int32 const &id,
                                      oatpp::String const &url) {
    auto status = StatusDto::createShared();
    status->status = "OK";
    status->code = 200;
    status->message = "User was successfully updated";
    return status;
  }

  oatpp::Object<StatusDto> deleteUser(oatpp::Int32 const &id) {
    auto status = StatusDto::createShared();
    status->status = "OK";
    status->code = 200;
    status->message = "User was successfully deleted";
    return status;
  }
  oatpp::Object<StatusDto> searchUser(oatpp::String const &url) {
    auto status = StatusDto::createShared();
    status->status = "OK";
    status->code = 200;
    status->message = "ID";
    return status;
  }
};

#endif