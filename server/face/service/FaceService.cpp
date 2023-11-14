/**
 * @file FaceService.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 人脸服务，提供人脸增删改查的接口。通过特征向量查询会在人脸特征库（内存中）查询，不会访问数据库。数据库的作
 * 用是持久化数据，人脸库的作用是加速查询。程序启动时会从数据库中读取数据，构建人脸库。
 * @version 0.1
 * @date 2023-11-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "FaceService.hpp"
#include "UserDb.hpp"
#include "UserDto.hpp"
#include "myBase64.hpp"
#include "networkUtils.hpp"
#include <cstdint>
#include <opencv2/core/hal/interface.h>
#include <vector>

namespace server::face {
constexpr float THRESHOLD = 0.35;

FaceService::FaceService() {
  // 从数据库中读取数据，构建人脸库
  auto results = getAllUsers();
  std::vector<long> ids;
  std::vector<std::vector<float>> features;
  for (auto &user : *results) {
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
  if (!ids.empty()) {
    // 从数据库中读取到数据，才会构建人脸库
    core::FaceLibraryManager::getInstance().createBatch(ids, features);
  }
}

// 辅助函数，处理批处理操作中的失败和错误ID，减少重复代码
void FaceService::handleBatchErrors(
    const std::vector<std::string> &errIdNumbers,
    const oatpp::Object<StatusDto> &status) {
  // 合并ID到一个字符串
  auto failed = joinIdNumbers(errIdNumbers);

  status->status = "Partial Content";
  status->code = 206; // HTTP状态码 206 Partial Content

  // 构建一个详细的状态消息
  std::string message = "Some users failed.";
  if (!errIdNumbers.empty()) {
    message += " Error IDs: " + failed + ".";
  }
  status->message = message;
}

// 辅助函数，批量算法调用，减少重复代码
void FaceService::batchInfer(std::vector<std::string> const &urls,
                             std::vector<std::string> const &idNumbers,
                             std::vector<std::vector<float>> &features,
                             std::vector<std::string> &errIdNumbers) {
  std::vector<std::future<bool>> futures;
  futures.reserve(urls.size());
  for (size_t i = 0; i < urls.size(); ++i) {
    futures.push_back(
        core::AlgoManager::getInstance().infer(urls[i], features[i]));
  }

  // 等待所有线程完成
  for (size_t i = 0; i < futures.size(); ++i) {
    if (!futures[i].get()) {
      errIdNumbers.push_back(idNumbers[i]);
    }
  }
}

// 提取特征向量
bool FaceService::extractFeature(std::string const &url,
                                 std::vector<float> &feature) {
  auto ret = core::AlgoManager::getInstance().infer(url, feature);
  return ret.get();
}

// 特征转base64
std::string FaceService::feature2base64(std::vector<float> &feature) {
  uchar *temp = reinterpret_cast<uchar *>(feature.data());
  std::vector<uchar> vec(temp, temp + feature.size() * sizeof(float));
  return flowengine::core::Base64::encode(vec);
}

oatpp::Int32 FaceService::getIdByIdNumber(
    oatpp::String const &idNumber,
    const oatpp::provider::ResourceHandle<oatpp::orm::Connection> &connection) {

  auto dbResult = m_database->getIdByIdNumber(idNumber, connection);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());
  OATPP_ASSERT_HTTP(dbResult->hasMoreToFetch(), Status::CODE_404,
                    "User not found");

  // 修改这里：使用 Vector 作为返回类型
  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  OATPP_ASSERT_HTTP(result && result->size() == 1, Status::CODE_500,
                    "Unknown error");

  // 获取 Vector 中的第一个元素，并提取 Int32 值
  return result[0]->id;
}

oatpp::String FaceService::getIdNumberById(
    oatpp::Int32 const &id,
    const oatpp::provider::ResourceHandle<oatpp::orm::Connection> &connection) {

  auto dbResult = m_database->getIdNumberById(id, connection);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());
  OATPP_ASSERT_HTTP(dbResult->hasMoreToFetch(), Status::CODE_404,
                    "User not found");

  // 修改这里：使用 Vector 作为返回类型
  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  OATPP_ASSERT_HTTP(result && result->size() == 1, Status::CODE_500,
                    "Unknown error");

  // 获取 Vector 中的第一个元素，并提取 String 值
  return result[0]->idNumber;
}

oatpp::Vector<oatpp::Object<UserDto>> FaceService::getAllUsers(
    const oatpp::provider::ResourceHandle<oatpp::orm::Connection> &connection) {

  auto dbResult = m_database->getAllUsers(connection);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  return result;
}

oatpp::Int32 FaceService::insertUser(
    std::string const &idNumber, std::string const &feature,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  auto user = UserDto::createShared();
  user->idNumber = idNumber;
  user->feature = feature;
  auto dbResult = m_database->createUser(user);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

  return oatpp::sqlite::Utils::getLastInsertRowId(dbResult->getConnection());
}

oatpp::Int32 FaceService::updateUserByIdNumber(
    std::string const &idNumber, std::string const &feature,
    oatpp::provider::ResourceHandle<oatpp::orm::Connection> const &connection) {
  // 获取id
  auto id = getIdByIdNumber(idNumber);
  auto user = UserDto::createShared();
  user->idNumber = idNumber;
  user->feature = feature;
  auto dbResult = m_database->updateUserById(id, user);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());
  return id;
}

oatpp::Object<StatusDto> FaceService::createUser(oatpp::String const &idNumber,
                                                 oatpp::String const &url) {
  auto status = StatusDto::createShared();
  std::vector<float> feature;
  // 提取特征向量
  if (!extractFeature(url, feature)) {
    status->status = "No Content";
    status->code = 204;
    status->message = "Face feature extraction failed.";
    return status;
  }

  // feature to base64
  std::string base64 = feature2base64(feature);

  auto id = insertUser(idNumber, base64);

  // 提取特征成功，接下来特征入库
  bool ok =
      core::FaceLibraryManager::getInstance().createOne(id, feature.data());
  if (!ok) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "User failed to create.";
  } else {
    status->status = "OK";
    status->code = 200;
    status->message = "User was successfully created.";
  }

  return status;
}

oatpp::Object<StatusDto>
FaceService::createUser(oatpp::Object<FaceDto> const &face) {
  // Post 本身只是解决了参数传递的问题，具体的逻辑还是调用Get的函数
  return createUser(face->userId, face->url);
}

oatpp::Object<StatusDto> FaceService::updateUser(oatpp::String const &idNumber,
                                                 oatpp::String const &url) {
  auto status = StatusDto::createShared();
  std::vector<float> feature;
  if (!extractFeature(url, feature)) {
    status->status = "No Content";
    status->code = 204;
    status->message = "Face feature extraction failed.";
    return status;
  }

  // feature to base64
  std::string base64 = feature2base64(feature);

  // 获取id
  auto id = updateUserByIdNumber(idNumber, base64);

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

// Post 更新
oatpp::Object<StatusDto>
FaceService::updateUser(oatpp::Object<FaceDto> const &face) {
  // Post 本身只是解决了参数传递的问题，具体的逻辑还是调用Get的函数
  return updateUser(face->userId, face->url);
}

oatpp::Object<StatusDto>
FaceService::deleteUser(oatpp::String const &idNumber) {

  auto status = StatusDto::createShared();

  // 根据idNumber删除数据库中的数据
  auto id = getIdByIdNumber(idNumber);
  auto dbResult = m_database->deleteUserById(id);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

  auto ret = core::FaceLibraryManager::getInstance().deleteOne(id);
  OATPP_ASSERT_HTTP(ret, Status::CODE_500, "Failed to delete the user.");

  status->status = "OK";
  status->code = 200;
  status->message = "User was successfully deleted.";
  return status;
}

oatpp::Object<StatusDto> FaceService::searchUser(oatpp::String const &url) {
  // 不涉及数据库操作，从人脸特征库中查询
  auto status = StatusDto::createShared();
  std::vector<float> feature;
  if (!extractFeature(url, feature)) {
    status->status = "No Content";
    status->code = 204;
    status->message = "Face feature extraction failed.";
    return status;
  }
  auto idx =
      core::FaceLibraryManager::getInstance().match(feature.data(), THRESHOLD);

  if (idx < 0) {
    status->status = "No Content";
    status->code = 204;
    status->message = "Face feature extraction failed.";
  } else {
    // 根据id查询idNumber
    status->status = "OK";
    status->code = 200;
    status->message = getIdNumberById(idx);
  }
  return status;
}

oatpp::Object<StatusDto>
FaceService::searchUser(oatpp::Object<ImageDto> const &images) {
  // Post 本身只是解决了参数传递的问题，具体的逻辑还是调用Get的函数
  return searchUser(images->url);
}

// 两图比对
oatpp::Object<StatusDto>
FaceService::compareTwoPictures(oatpp::String const &url1,
                                oatpp::String const &url2) {
  auto status = StatusDto::createShared();
  std::vector<float> feature1;
  std::vector<float> feature2;
  // 这里不调用extractFeature函数，是因为需要并发提取特征
  auto ret1 = core::AlgoManager::getInstance().infer(url1, feature1);
  auto ret2 = core::AlgoManager::getInstance().infer(url2, feature2);

  if (!ret1.get() || !ret2.get()) {
    status->status = "No Content";
    status->code = 204;
    status->message = "Face feature extraction failed.";
    return status;
  }
  // 算法提取数据后都是归一化的，所以直接点积即可
  auto score = infer::utils::dot(feature1, feature2);

  status->status = "OK";
  status->code = 200;
  // 比较阈值
  if (score < THRESHOLD) {
    status->message = "0";
  } else {
    status->message = "1";
  }
  return status;
}

// Post 两图比对
oatpp::Object<StatusDto> FaceService::compareTwoPictures(
    oatpp::Vector<oatpp::Object<ImageDto>> const &images) {

  OATPP_ASSERT_HTTP(images->size() == 2, Status::CODE_500,
                    "The number of images entered must be two!");

  // Post 本身只是解决了参数传递的问题，具体的逻辑还是调用Get的函数
  return compareTwoPictures(images[0]->url, images[1]->url);
}

// 批量新增
oatpp::Object<StatusDto>
FaceService::createBatch(oatpp::Vector<oatpp::Object<FaceDto>> const &users) {
  auto status = StatusDto::createShared();

  std::vector<std::string> IdNumbers;
  std::vector<std::string> urls;
  for (size_t i = 0; i < users->size(); ++i) {
    IdNumbers.push_back(users[i]->userId);
    urls.push_back(users[i]->url);
  }

  // 特征向量集合
  std::vector<std::vector<float>> features(users->size());
  std::vector<std::string> errIdNumbers;
  std::vector<std::string> vecsBase64; // 用于存储提取成功的特征向量的base64
  batchInfer(urls, IdNumbers, features, errIdNumbers);

  // 从IdNumbers中移除失败的id，顺带移除features中的失败特征向量
  for (size_t i = 0; i < errIdNumbers.size(); ++i) {
    IdNumbers.erase(
        std::remove(IdNumbers.begin(), IdNumbers.end(), errIdNumbers[i]),
        IdNumbers.end());
    features.erase(features.begin() + i);
  }

  // 全部失败的情况
  if (IdNumbers.empty()) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "All users failed to create.";
    return status;
  }

  // 此时的IdNumbers与features一一对应
  std::vector<long> ids;
  // 入库
  for (size_t i = 0; i < IdNumbers.size(); ++i) {
    auto id = insertUser(IdNumbers[i], feature2base64(features[i]));
    ids.push_back(id);
  }

  // 更新人脸库
  core::FaceLibraryManager::getInstance().createBatch(ids, features);
  // 处理批处理中的错误入库的情况
  handleBatchErrors(errIdNumbers, status);
  return status;
}

// 批量更新
oatpp::Object<StatusDto>
FaceService::updateBatch(oatpp::Vector<oatpp::Object<FaceDto>> const &users) {
  auto status = StatusDto::createShared();

  std::vector<std::string> IdNumbers;
  std::vector<std::string> urls;
  for (size_t i = 0; i < users->size(); ++i) {
    IdNumbers.push_back(users[i]->userId);
    urls.push_back(users[i]->url);
  }

  // 特征向量集合
  std::vector<std::vector<float>> features(users->size());
  std::vector<std::string> errIdNumbers;
  std::vector<std::string> vecsBase64; // 用于存储提取成功的特征向量的base64
  batchInfer(urls, IdNumbers, features, errIdNumbers);

  // 从IdNumbers中移除失败的id，顺带移除features中的失败特征向量
  for (size_t i = 0; i < errIdNumbers.size(); ++i) {
    IdNumbers.erase(
        std::remove(IdNumbers.begin(), IdNumbers.end(), errIdNumbers[i]),
        IdNumbers.end());
    features.erase(features.begin() + i);
  }

  // 全部失败的情况
  if (IdNumbers.empty()) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "All users failed to update.";
    return status;
  }

  // 此时的IdNumbers与features一一对应
  std::vector<long> ids;
  // 入库
  for (size_t i = 0; i < IdNumbers.size(); ++i) {
    auto id = updateUserByIdNumber(IdNumbers[i], feature2base64(features[i]));
    ids.push_back(id);
  }

  // 更新人脸库
  core::FaceLibraryManager::getInstance().updateBatch(ids, features);
  // 处理批处理中的错误入库的情况
  handleBatchErrors(errIdNumbers, status);
  return status;
}

// 批量删除
oatpp::Object<StatusDto>
FaceService::deleteBatch(oatpp::Vector<oatpp::Object<FaceDto>> const &users) {
  auto status = StatusDto::createShared();

  std::vector<std::string> IdNumbers;
  for (size_t i = 0; i < users->size(); ++i) {
    IdNumbers.push_back(users[i]->userId);
  }

  // 利用IdNumbers查询id
  oatpp::Vector<oatpp::Int32> idsDB;
  for (size_t i = 0; i < IdNumbers.size(); ++i) {
    auto id = getIdByIdNumber(IdNumbers[i]);
    idsDB->push_back(id);
  }

  std::vector<long> ids;
  for (size_t i = 0; i < idsDB->size(); ++i) {
    ids.push_back(idsDB[i]);
  }

  // 删除数据库中的数据
  auto dbResult = m_database->deleteUsersByIds(idsDB);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

  // 删除人脸库中的数据
  core::FaceLibraryManager::getInstance().deleteBatch(ids);

  status->status = "OK";
  status->code = 200;
  status->message = "Users were successfully deleted.";
  return status;
}
} // namespace server::face