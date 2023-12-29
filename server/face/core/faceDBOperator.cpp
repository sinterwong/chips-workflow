/**
 * @file faceDBOperator.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-29
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "faceDBOperator.hpp"
#include "common/myBase64.hpp"
#include "faceLibManager.hpp"
#include <memory>

namespace server::face::core {

FaceDBOperator *FaceDBOperator::instance = nullptr;

void FaceDBOperator::getIdsAndFeatures(
    std::vector<long> &ids, std::vector<std::vector<float>> &features,
    oatpp::Vector<oatpp::Object<UserDto>> const &users) {
  for (auto &user : *users) {
    auto id = user->id;
    auto feature = user->feature;
    auto ret = flowengine::core::Base64::decode(feature);
    // 确保解码后的字符串大小是浮点数大小的整数倍
    if (ret.size() % sizeof(float) != 0) {
      throw std::runtime_error(
          "Decoded string size is not a multiple of the float size.");
    }
    std::vector<float> floatVector(ret.size() / sizeof(float));
    std::memcpy(floatVector.data(), ret.data(), ret.size());
    ids.push_back(id);
    features.push_back(std::move(floatVector));
  }
}

bool FaceDBOperator::restoreFacelib(std::string const &libName,
                                    oatpp::Object<StatusDto> &status,
                                    bool needCreate) {
  // 获取所有id和feature
  auto results = getUsersByLibName(libName, status);
  if (!results) {
    if (needCreate) {
      // 人脸库中没有数据，需要创建
      return core::FaceLibraryManager::getInstance().registerFacelib(libName);
    }
  } else {
    // 人脸库中有数据，需要恢复
    std::vector<long> ids;
    std::vector<std::vector<float>> features;
    // 获取ids和features
    getIdsAndFeatures(ids, features, results);
    return core::FaceLibraryManager::getInstance().registerFacelib(libName, ids,
                                                                   features);
  }
  return false;
}

oatpp::Int32 FaceDBOperator::getIdByIdNumber(
    oatpp::String const &idNumber, oatpp::Object<StatusDto> &status,
    const oatpp::provider::ResourceHandle<oatpp::orm::Connection> &connection) {

  auto dbResult = m_database->getIdByIdNumber(idNumber, connection);
  // OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
  //                   dbResult->getErrorMessage());
  if (!dbResult->isSuccess()) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = dbResult->getErrorMessage();
    return -1;
  }
  // OATPP_ASSERT_HTTP(dbResult->hasMoreToFetch(), Status::CODE_404,
  //                   "User not found");
  if (!dbResult->hasMoreToFetch()) {
    status->code = 404;
    status->status = "Not Found";
    status->message = "User not found";
    return -1;
  }

  // 修改这里：使用 Vector 作为返回类型
  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  // OATPP_ASSERT_HTTP(result && result->size() == 1, Status::CODE_500,
  //                   "Unknown error");
  if (!result || result->size() != 1) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = "Unknown error";
    return -1;
  }

  // 获取 Vector 中的第一个元素，并提取 Int32 值
  return result[0]->id;
}

oatpp::String FaceDBOperator::getIdNumberById(
    oatpp::Int32 const &id, oatpp::Object<StatusDto> &status,
    const oatpp::provider::ResourceHandle<oatpp::orm::Connection> &connection) {

  auto dbResult = m_database->getIdNumberById(id, connection);
  // OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
  //                   dbResult->getErrorMessage());

  if (!dbResult->isSuccess()) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = dbResult->getErrorMessage();
    return "";
  }
  // OATPP_ASSERT_HTTP(dbResult->hasMoreToFetch(), Status::CODE_404,
  //                   "User not found");

  if (!dbResult->hasMoreToFetch()) {
    status->code = 404;
    status->status = "Not Found";
    status->message = "User not found";
    return "";
  }

  // 修改这里：使用 Vector 作为返回类型
  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  // OATPP_ASSERT_HTTP(result && result->size() == 1, Status::CODE_500,
  //                   "Unknown error");

  if (!result || result->size() != 1) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = "Unknown error";
    return "";
  }

  // 获取 Vector 中的第一个元素，并提取 String 值
  return result[0]->idNumber;
}

oatpp::Vector<oatpp::Object<UserDto>> FaceDBOperator::getAllUsers(
    const oatpp::provider::ResourceHandle<oatpp::orm::Connection> &connection) {

  auto dbResult = m_database->getAllUsers(connection);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  return result;
}

oatpp::String FaceDBOperator::getLibNameById(
    oatpp::Int32 const &id, oatpp::Object<StatusDto> &status,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  auto dbResult = m_database->getLibNameById(id, connection);
  if (!dbResult->isSuccess()) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = dbResult->getErrorMessage();
    return "";
  }
  if (!dbResult->hasMoreToFetch()) {
    status->code = 404;
    status->status = "Not Found";
    status->message = "User not found";
    return "";
  }
  // 修改这里：使用 Vector 作为返回类型
  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  if (!result || result->size() != 1) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = "Unknown error";
    return "";
  }

  // 获取 Vector 中的第一个元素，并提取 String 值
  return result[0]->libName;
}

oatpp::Vector<oatpp::Object<UserDto>> FaceDBOperator::getUsersByLibName(
    oatpp::String const &libName, oatpp::Object<StatusDto> &status,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  auto dbResult = m_database->getUsersByLibName(libName, connection);
  if (!dbResult->isSuccess()) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = dbResult->getErrorMessage();
    return nullptr;
  }
  if (!dbResult->hasMoreToFetch()) {
    status->code = 404;
    status->status = "Not Found";
    status->message = "User not found";
    return nullptr;
  }
  // 修改这里：使用 Vector 作为返回类型
  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  return result;
}

oatpp::String FaceDBOperator::getLibNameByIdNumber(
    oatpp::String const &idNumber, oatpp::Object<StatusDto> &status,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  auto dbResult = m_database->getLibNameByIdNumber(idNumber, connection);
  if (!dbResult->isSuccess()) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = dbResult->getErrorMessage();
    return "";
  }
  if (!dbResult->hasMoreToFetch()) {
    status->code = 404;
    status->status = "Not Found";
    status->message = "User not found";
    return "";
  }
  // 修改这里：使用 Vector 作为返回类型
  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  if (!result || result->size() != 1) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = "Unknown error";
    return "";
  }

  // 获取 Vector 中的第一个元素，并提取 String 值
  return result[0]->libName;
}

oatpp::Int32 FaceDBOperator::insertUser(
    std::string const &idNumber, std::string const &libName,
    std::string const &feature, oatpp::Object<StatusDto> &status,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  auto user = UserDto::createShared();
  user->idNumber = idNumber;
  user->libName = libName;
  user->feature = feature;
  auto dbResult = m_database->createUser(user);
  if (!dbResult->isSuccess()) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = dbResult->getErrorMessage();
    return -1;
  }

  return oatpp::sqlite::Utils::getLastInsertRowId(dbResult->getConnection());
}

oatpp::Int32 FaceDBOperator::updateUserByIdNumber(
    std::string const &idNumber, std::string const &libName,
    std::string const &feature, oatpp::Object<StatusDto> &status,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  // 获取id
  auto id = getIdByIdNumber(idNumber, status, connection);

  if (id < 0) {
    return -1;
  }

  auto user = UserDto::createShared();
  user->idNumber = idNumber;
  user->feature = feature;
  user->libName = libName;
  auto dbResult = m_database->updateUserById(id, user);
  // OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
  //                   dbResult->getErrorMessage());
  if (!dbResult->isSuccess()) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = dbResult->getErrorMessage();
    return -1;
  }
  return id;
}

oatpp::Int32 FaceDBOperator::deleteUserByIdNumber(
    std::string const &idNumber, oatpp::Object<StatusDto> &status,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  // 获取id
  auto id = getIdByIdNumber(idNumber, status, connection);
  if (id < 0) {
    return -1;
  }
  auto dbResult = m_database->deleteUserById(id);
  // OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
  //                   dbResult->getErrorMessage());
  if (!dbResult->isSuccess()) {
    status->code = 500;
    status->status = "Internal Server Error";
    status->message = dbResult->getErrorMessage();
    return -1;
  }
  return id;
}

std::shared_ptr<oatpp::orm::QueryResult> FaceDBOperator::executeSql(
    oatpp::String const &sql,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  auto dbResult = m_database->executeQuery(sql, {});
  return dbResult;
}
} // namespace server::face::core