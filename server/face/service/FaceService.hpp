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
#include "algoManager.hpp"
#include "faceLibManager.hpp"
#include <oatpp/web/protocol/http/Http.hpp>
#include <string>

#ifndef __CRUD_FACE_SERVICE_HPP_
#define __CRUD_FACE_SERVICE_HPP_
namespace server::face {
class FaceService {
private:
  using Status = oatpp::web::protocol::http::Status;

public:
  oatpp::Object<StatusDto> createUser(oatpp::Int32 const &id,
                                      oatpp::String const &url) {
    auto status = StatusDto::createShared();
    std::vector<float> feature;
    auto ret = core::AlgoManager::getInstance().infer(url, feature);

    if (!ret.get()) {
      status->status = "Service Unavailable";
      status->code = 503;
      status->message = "Face feature extraction failed.";
      return status;
    }

    // 提取特征成功，接下来特征入库
    bool ok =
        core::FaceLibraryManager::getInstance().createOne(id, feature.data());

    if (!ok) {
      status->status = "Service Unavailable";
      status->code = 503;
      status->message = "User failed to enter database.";
    } else {
      status->status = "OK";
      status->code = 200;
      status->message = "User was successfully created.";
    }

    return status;
  }

  oatpp::Object<StatusDto> updateUser(oatpp::Int32 const &id,
                                      oatpp::String const &url) {
    auto status = StatusDto::createShared();
    std::vector<float> feature;
    auto ret = core::AlgoManager::getInstance().infer(url, feature);

    if (!ret.get()) {
      status->status = "Service Unavailable";
      status->code = 503;
      status->message = "Face feature extraction failed.";
      return status;
    }

    // 提取特征成功，接下来特征入库
    bool ok =
        core::FaceLibraryManager::getInstance().updateOne(id, feature.data());
    if (!ok) {
      status->status = "Service Unavailable";
      status->code = 503;
      status->message = "User failed to update";
    } else {
      status->status = "OK";
      status->code = 200;
      status->message = "User was successfully updated.";
    }

    return status;
  }

  oatpp::Object<StatusDto> deleteUser(oatpp::Int32 const &id) {

    auto status = StatusDto::createShared();

    auto ret = core::FaceLibraryManager::getInstance().deleteOne(id);
    if (!ret) {
      status->status = "Service Unavailable";
      status->code = 503;
      status->message = "Failed to delete the user.";
    } else {
      status->status = "OK";
      status->code = 200;
      status->message = "User was successfully deleted.";
    }
    return status;
  }

  oatpp::Object<StatusDto> searchUser(oatpp::String const &url) {
    auto status = StatusDto::createShared();
    std::vector<float> feature;
    auto ret = core::AlgoManager::getInstance().infer(url, feature);

    if (!ret.get()) {
      status->status = "Service Unavailable";
      status->code = 503;
      status->message = "Face feature extraction failed.";
      return status;
    }
    auto idx =
        core::FaceLibraryManager::getInstance().match(feature.data(), 0.2);
    if (idx < 0) {
      status->status = "No Content";
      status->code = 204;
      status->message = "No face found";
    } else {
      status->status = "OK";
      status->code = 200;
      status->message = std::to_string(idx);
    }
    return status;
  }
};
} // namespace server::face
#endif