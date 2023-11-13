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
#include "networkUtils.hpp"
#include <cstdint>
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
    std::string decodedString;
    base64_decode(feature->c_str(), decodedString);
    // 将特征向量转换为float数组
    // 确保解码后的字符串大小是浮点数大小的整数倍
    if (decodedString.size() % sizeof(float) != 0) {
      throw std::runtime_error(
          "Decoded string size is not a multiple of the float size.");
    }

    std::vector<float> floatVector(decodedString.size() / sizeof(float));
    std::memcpy(floatVector.data(), decodedString.data(), decodedString.size());
    ids.push_back(id);
    features.push_back(std::move(floatVector));
  }
  if (!ids.empty()) {
    // 从数据库中读取到数据，才会构建人脸库
    core::FaceLibraryManager::getInstance().loadFacelib(ids, features);
  }
}

// 辅助函数，处理批处理操作中的失败和错误ID，减少重复代码
void FaceService::handleBatchErrors(const std::vector<long> &failed_ids,
                                    const std::vector<long> &err_ids,
                                    const oatpp::Object<StatusDto> &status) {
  // 合并ID到一个字符串
  auto failed = joinIds(failed_ids);
  auto errored = joinIds(err_ids);

  status->status = "Partial Content";
  status->code = 206; // HTTP状态码 206 Partial Content

  // 构建一个详细的状态消息
  std::string message = "Some users failed.";
  if (!failed_ids.empty()) {
    message += " Algorithm failed IDs: " + failed + ".";
  }
  if (!err_ids.empty()) {
    message += " Facelib failed IDs: " + errored + ".";
  }
  status->message = message;
}

// 辅助函数，批量算法调用，减少重复代码
void FaceService::batchInfer(const std::vector<std::string> &urls,
                             std::vector<float *> &vecs,
                             std::vector<std::vector<float>> &features,
                             std::vector<long> &failed_ids) {
  std::vector<std::future<bool>> futures;
  futures.reserve(urls.size());
  for (size_t i = 0; i < urls.size(); ++i) {
    futures.push_back(
        core::AlgoManager::getInstance().infer(urls[i], features[i]));
  }

  // 等待所有线程完成
  for (size_t i = 0; i < futures.size(); ++i) {
    if (!futures[i].get()) {
      failed_ids.push_back(i);

    } else {
      // 将成功的特征向量指针加入vecs
      vecs.push_back(features[i].data());
    }
  }
}

oatpp::Object<StatusDto> FaceService::createUser(oatpp::String const &idNumber,
                                                 oatpp::String const &url) {
  auto status = StatusDto::createShared();
  std::vector<float> feature;
  auto ret = core::AlgoManager::getInstance().infer(url, feature);

  OATPP_ASSERT_HTTP(ret.get(), Status::CODE_500,
                    "Face feature extraction failed.");

  // feature to base64
  uchar *temp = reinterpret_cast<uchar *>(feature.data());
  std::string base64;
  base64_encode(temp, feature.size() * sizeof(float), base64);

  auto user = UserDto::createShared();
  user->idNumber = idNumber;
  user->feature = base64;

  auto dbResult = m_database->createUser(user);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

  auto id = oatpp::sqlite::Utils::getLastInsertRowId(dbResult->getConnection());

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

oatpp::Object<StatusDto> FaceService::updateUser(oatpp::String const &idNumber,
                                                 oatpp::String const &url) {
  auto status = StatusDto::createShared();
  std::vector<float> feature;
  auto ret = core::AlgoManager::getInstance().infer(url, feature);

  OATPP_ASSERT_HTTP(ret.get(), Status::CODE_500,
                    "Face feature extraction failed.");

  // feature to base64
  uchar *temp = reinterpret_cast<uchar *>(feature.data());
  std::string base64;
  base64_encode(temp, feature.size() * sizeof(float), base64);

  // 获取id
  auto id = getIdByIdNumber(idNumber);
  auto user = UserDto::createShared();
  user->idNumber = idNumber;
  user->feature = base64;
  m_database->updateUserById(id, user);

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

oatpp::Object<StatusDto>
FaceService::deleteUser(oatpp::String const &idNumber) {

  auto status = StatusDto::createShared();

  // 根据idNumber删除数据库中的数据
  auto id = getIdByIdNumber(idNumber);
  auto dbResult = m_database->deleteUserById(id);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

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

oatpp::Vector<oatpp::Object<UserDto>> FaceService::getAllUsers(
    const oatpp::provider::ResourceHandle<oatpp::orm::Connection> &connection) {

  auto dbResult = m_database->getAllUsers(connection);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

  auto result = dbResult->fetch<oatpp::Vector<oatpp::Object<UserDto>>>();
  return result;
}

oatpp::Object<StatusDto> FaceService::searchUser(oatpp::String const &url) {
  // 不涉及数据库操作，从人脸特征库中查询
  auto status = StatusDto::createShared();
  std::vector<float> feature;
  auto ret = core::AlgoManager::getInstance().infer(url, feature);

  if (!ret.get()) {
    status->status = "Service Unavailable";
    status->code = 500;
    status->message = "Face feature extraction failed.";
    return status;
  }
  auto idx =
      core::FaceLibraryManager::getInstance().match(feature.data(), THRESHOLD);
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

// 两图比对
oatpp::Object<StatusDto>
FaceService::compareTwoPictures(oatpp::String const &url1,
                                oatpp::String const &url2) {
  auto status = StatusDto::createShared();
  std::vector<float> feature1;
  std::vector<float> feature2;
  auto ret1 = core::AlgoManager::getInstance().infer(url1, feature1);
  auto ret2 = core::AlgoManager::getInstance().infer(url2, feature2);

  if (!ret1.get() || !ret2.get()) {
    status->status = "Service Unavailable";
    status->code = 503;
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

/* 人脸逻辑改变较大，暂时不实现批处理
// 批量新增
oatpp::Object<StatusDto>
FaceService::createBatch(const oatpp::Object<FacelibDto> &users) {
  auto status = StatusDto::createShared();
  if (!users || !users->ids || !users->urls ||
      users->ids->size() != users->urls->size()) {
    // 返回错误状态，表示输入数据有问题
    status->status = "Bad Request";
    status->code = 400;
    status->message = "Invalid input data.";
    return status;
  }

  std::vector<long> ids(users->ids->begin(), users->ids->end());
  std::vector<std::string> urls(users->urls->begin(), users->urls->end());

  std::vector<std::vector<float>> features;
  // vecs.size() + failed_ids.size() == ids.size()
  std::vector<float *> vecs;    // 用于存储提取成功的特征向量指针
  std::vector<long> failed_ids; // 用于存储失败的id
  // 批量推理
  batchInfer(urls, vecs, features, failed_ids);

  // 从ids中移除失败的id
  ids.erase(std::remove_if(ids.begin(), ids.end(),
                           [&](long id) {
                             return std::find(failed_ids.begin(),
                                              failed_ids.end(),
                                              id) != failed_ids.end();
                           }),
            ids.end());

  if (ids.empty()) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "All users failed to create.";
    return status;
  }
  // 此时的ids与vecs一一对应，可以直接传入
  std::vector<long> err_ids;
  core::FaceLibraryManager::getInstance().createBatch(ids, vecs.data(),
                                                      err_ids);
  // 处理批处理中的错误入库的情况
  handleBatchErrors(failed_ids, err_ids, status);
  return status;
}

// 批量更新
oatpp::Object<StatusDto>
FaceService::updateBatch(oatpp::Object<FacelibDto> const &users) {
  auto status = StatusDto::createShared();
  if (!users || !users->ids || !users->urls ||
      users->ids->size() != users->urls->size()) {
    // 返回错误状态，表示输入数据有问题
    status->status = "Bad Request";
    status->code = 400;
    status->message = "Invalid input data.";
    return status;
  }
  std::vector<long> ids(users->ids->begin(), users->ids->end());
  std::vector<std::string> urls(users->urls->begin(), users->urls->end());

  std::vector<std::vector<float>> features;
  // vecs.size() + failed_ids.size() == ids.size()
  std::vector<float *> vecs;    // 用于存储提取成功的特征向量指针
  std::vector<long> failed_ids; // 用于存储失败的id
  // 批量推理
  batchInfer(urls, vecs, features, failed_ids);

  // 从ids中移除失败的id
  ids.erase(std::remove_if(ids.begin(), ids.end(),
                           [&](long id) {
                             return std::find(failed_ids.begin(),
                                              failed_ids.end(),
                                              id) != failed_ids.end();
                           }),
            ids.end());

  if (ids.empty()) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "All users failed to update.";
    return status;
  }

  std::vector<long> err_ids;
  core::FaceLibraryManager::getInstance().updateBatch(ids, vecs.data(),
                                                      err_ids);

  handleBatchErrors(failed_ids, err_ids, status);
  return status;
}

// 批量删除
oatpp::Object<StatusDto>
FaceService::deleteBatch(oatpp::Object<FacelibDto> const &users) {
  auto status = StatusDto::createShared();
  if (!users || !users->ids) {
    // 返回错误状态，表示输入数据有问题
    status->status = "Bad Request";
    status->code = 400;
    status->message = "Invalid input data.";
    return status;
  }

  std::vector<long> ids(users->ids->begin(), users->ids->end());
  std::vector<long> err_ids;
  core::FaceLibraryManager::getInstance().deleteBatch(ids, err_ids);
  handleBatchErrors({}, err_ids, status);
  return status;
}
*/
} // namespace server::face