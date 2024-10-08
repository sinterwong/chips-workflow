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
#include "StatusDto.hpp"
#include "UserDto.hpp"
#include "faceQuality.hpp"
#include "faceRecognition.hpp"
#include "networkUtils.hpp"
#include <cassert>
#include <cstdint>
#include <opencv2/core/hal/interface.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "faceDBOperator.hpp"

namespace server::face {

constexpr float THRESHOLD = 0.35;

using core::FaceDBOperator;

using Status = oatpp::web::protocol::http::Status;

// 辅助函数，处理批处理操作中的失败和错误ID，减少重复代码
void FaceService::handleBatchErrors(
    std::vector<std::string> const &errIdNumbersAlgo,
    std::vector<std::string> const &errIdNumbersDB,
    const oatpp::Object<StatusDto> &status) {

  if (errIdNumbersAlgo.empty() && errIdNumbersDB.empty()) {
    // 没有错误
    status->status = "OK";
    status->code = 200;
    status->message = "All users were successfully created.";
    return;
  }
  // 合并ID到一个字符串
  auto algoFailed = joinIdNumbers(errIdNumbersAlgo);
  auto dbFailed = joinIdNumbers(errIdNumbersDB);
  status->status = "Partial Content";
  status->code = 206; // HTTP状态码 206 Partial Content

  // 构建一个详细的状态消息
  std::string message = "Some users failed.\n";
  if (!errIdNumbersAlgo.empty()) {
    message += "Algorithm failed: ";
    message += algoFailed + "\n";
  }
  if (!errIdNumbersDB.empty()) {
    message += "Database failed: ";
    message += dbFailed + "\n";
  }
  status->message = message;
}

// 辅助函数，批量算法调用，减少重复代码
void FaceService::batchInfer(std::vector<std::string> const &urls,
                             std::vector<std::string> const &idNumbers,
                             std::vector<std::vector<float>> &features,
                             std::vector<std::string> &errIdNumbers) {
  for (size_t i = 0; i < urls.size(); ++i) {
    if (!extractFeature(urls[i], features[i])) {
      errIdNumbers.push_back(idNumbers[i]);
    }
  }
}

// 提取特征向量
bool FaceService::extractFeature(std::string const &url,
                                 std::vector<float> &feature) {
  auto ret = core::FaceRecognition::getInstance().extract(url, feature);
  return ret;
}

// 特征转base64
std::string FaceService::feature2base64(std::vector<float> &feature) {
  uchar *temp = reinterpret_cast<uchar *>(feature.data());
  std::vector<uchar> vec(temp, temp + feature.size() * sizeof(float));
  return utils::Base64::encode(vec);
}

oatpp::Object<StatusDto> FaceService::createUser(oatpp::String const &idNumber,
                                                 oatpp::String const &libName,
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

  auto id = FaceDBOperator::getInstance().insertUser(idNumber, libName, base64,
                                                     status);
  if (id < 0) {
    return status;
  }

  // 入库成功，备份人脸图片并且更新人脸库
  backupImage(id, libName, idNumber, url);

  // 检查人脸库是否在线
  if (core::FaceLibraryManager::getInstance().CHECK_FACELIB_EXIST(libName)) {
    // 如果人脸库在线，则需要更新人脸库，否则不需要
    core::FaceLibraryManager::getInstance().createOne(libName, id,
                                                      feature.data());
  }
  status->status = "OK";
  status->code = 200;
  status->message = "User was successfully created.";
  return status;
}

oatpp::Object<StatusDto>
FaceService::createUser(oatpp::Object<FaceDto> const &face) {
  // Post 本身只是解决了参数传递的问题，具体的逻辑还是调用Get的函数
  return createUser(face->userId, face->libName, face->url);
}

oatpp::Object<StatusDto> FaceService::updateUser(oatpp::String const &idNumber,
                                                 oatpp::String const &libName,
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

  // 更新并获取id
  auto id = FaceDBOperator::getInstance().updateUserByIdNumber(
      idNumber, libName, base64, status);
  if (id < 0) {
    return status;
  }

  /* 至此，数据库中的数据已经更新，接下来更新人脸库中的数据
   * 考虑一个情况，如果一条数据在本次更新中更新了一下libName字段，id和idNumber没有发生改变，
   * 这步操作对应了人脸库中的”换库“，如果原来的库在线，需要先删除原来库中的数据，不在线则不需要
   */

  // 入库成功，备份人脸图片并且更新人脸库
  backupImage(id, libName, idNumber, url);

  // 检查人脸库是否存在
  if (core::FaceLibraryManager::getInstance().CHECK_FACELIB_EXIST(libName)) {
    // 人脸库在线，需要进行更新操作
    // 需要知道libName较上次是否发生了改变，如果发生了变化需要先删除原来库中的数据
    auto dbLibName =
        FaceDBOperator::getInstance().getLibNameByIdNumber(idNumber, status);
    if (dbLibName->empty()) {
      return status;
    }

    // 如果原来的库在线，需要删除原来库中的数据
    bool needDelete = true ? dbLibName != libName : false;

    // 如果是“换库”操作，这里的updateOne实际上执行的是createOne操作
    core::FaceLibraryManager::getInstance().updateOne(libName, id,
                                                      feature.data());
    if (needDelete) {
      // 如果原来的库在线，需要删除原来库中的数据
      if (core::FaceLibraryManager::getInstance().CHECK_FACELIB_EXIST(
              dbLibName)) {
        core::FaceLibraryManager::getInstance().deleteOne(dbLibName, id);
      }
    }
  }
  status->status = "OK";
  status->code = 200;
  status->message = "User was successfully updated.";
  return status;
}

// Post 更新
oatpp::Object<StatusDto>
FaceService::updateUser(oatpp::Object<FaceDto> const &face) {
  // Post 本身只是解决了参数传递的问题，具体的逻辑还是调用Get的函数
  return updateUser(face->userId, face->libName, face->url);
}

oatpp::Object<StatusDto>
FaceService::deleteUser(oatpp::String const &idNumber) {

  auto status = StatusDto::createShared();

  auto libName =
      FaceDBOperator::getInstance().getLibNameByIdNumber(idNumber, status);

  if (libName->empty()) {
    return status;
  }

  // 根据idNumber删除数据库中的数据
  auto id =
      FaceDBOperator::getInstance().deleteUserByIdNumber(idNumber, status);

  if (id < 0) {
    return status;
  }

  // 检查人脸库是否在线
  if (core::FaceLibraryManager::getInstance().CHECK_FACELIB_EXIST(libName)) {
    // 人脸库在线，需要进行更新操作
    auto ret = core::FaceLibraryManager::getInstance().deleteOne(libName, id);
    OATPP_ASSERT_HTTP(ret, Status::CODE_500, "Failed to delete the user.");
  }

  status->status = "OK";
  status->code = 200;
  status->message = "User was successfully deleted.";
  return status;
}

oatpp::Object<StatusDto> FaceService::searchUser(oatpp::String const &libName,
                                                 oatpp::String const &url) {
  // 不涉及数据库操作，从人脸特征库中查询
  auto status = StatusDto::createShared();

  if (!core::FaceLibraryManager::getInstance().CHECK_FACELIB_EXIST(libName)) {
    // 人脸库不存在，需要尝试恢复
    if (!FaceDBOperator::getInstance().restoreFacelib(libName, status)) {
      // 恢复失败，返回错误
      status->status = "Not Found";
      status->code = 404;
      status->message = "Facelib not found.";
      return status;
    }
    // 恢复成功，继续执行
  }

  std::vector<float> feature;
  if (!extractFeature(url, feature)) {
    status->status = "No Content";
    status->code = 204;
    status->message = "Face feature extraction failed.";
    return status;
  }
  auto idx = core::FaceLibraryManager::getInstance().match(
      libName, feature.data(), THRESHOLD);

  if (idx < 0) {
    status->status = "No Content";
    status->code = 204;
    status->message = "User not found.";
  } else {
    auto ret = FaceDBOperator::getInstance().getIdNumberById(idx, status);
    if (!ret->empty()) {
      status->status = "OK";
      status->code = 200;
      status->message = ret;
    }
  }
  return status;
}

oatpp::Object<StatusDto>
FaceService::searchUser(oatpp::Object<ImageDto> const &images) {
  // Post 本身只是解决了参数传递的问题，具体的逻辑还是调用Get的函数
  return searchUser(images->name, images->url);
}

// 两图比对
oatpp::Object<StatusDto>
FaceService::compareTwoPictures(oatpp::String const &url1,
                                oatpp::String const &url2) {
  auto status = StatusDto::createShared();
  std::vector<float> feature1;
  std::vector<float> feature2;
  // 这里不调用extractFeature函数，是因为需要并发提取特征
  auto ret1 = extractFeature(url1, feature1);
  auto ret2 = extractFeature(url2, feature2);

  if (!ret1 || !ret2) {
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

  // set是无序的，需要保证顺序，所以使用vector
  std::vector<std::string> idNumbers;
  std::vector<std::string> libNames;
  std::vector<std::string> urls;
  for (size_t i = 0; i < users->size(); ++i) {
    idNumbers.push_back(users[i]->userId);
    urls.push_back(users[i]->url);
    libNames.push_back(users[i]->libName);
  }

  // 特征向量集合
  std::vector<std::vector<float>> features(users->size());

  // 算法执行失败的idNumber
  std::vector<std::string> erridNumbersAlgo;

  // 用于存储提取成功的特征向量的base64
  std::vector<std::string> vecsBase64;

  // 批量提取特征向量
  batchInfer(urls, idNumbers, features, erridNumbersAlgo);

  // 从idNumbers中移除失败的id，顺带移除features中的失败特征向量
  for (size_t i = 0; i < erridNumbersAlgo.size(); ++i) {
    // idNumbers 和 features
    // 一一对应，要获取idNumbers中的索引，再移除features中的特征向量 获取索引
    auto index = std::distance(
        idNumbers.begin(),
        std::find(idNumbers.begin(), idNumbers.end(), erridNumbersAlgo[i]));
    idNumbers.erase(idNumbers.begin() + index);
    libNames.erase(libNames.begin() + index);
    features.erase(features.begin() + index);
    urls.erase(urls.begin() + index);
  }

  // 此时的idNumbers与features与libNames一一对应
  assert(idNumbers.size() == features.size() &&
         idNumbers.size() == libNames.size() &&
         idNumbers.size() == urls.size());

  // 全部失败的情况
  if (idNumbers.empty()) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "All users failed to create.";
    return status;
  }

  // 数据库插入失败的情况
  std::vector<std::string> erridNumbersDB;

  // 更新人脸库时成功的id
  std::vector<long> ids;

  // 插入数据库
  for (size_t i = 0; i < idNumbers.size(); ++i) {
    auto id = FaceDBOperator::getInstance().insertUser(
        idNumbers[i], libNames[i], feature2base64(features[i]), status);
    if (id < 0) {
      // 更新数据库时失败项
      erridNumbersDB.push_back(idNumbers[i]);
    } else {
      ids.push_back(id);
    }
  }

  // 插入数据库失败的情况需要同步给features
  for (size_t i = 0; i < erridNumbersDB.size(); ++i) {
    // 从idNumbers中的索引，再移除features中的特征向量
    auto index = std::distance(
        idNumbers.begin(),
        std::find(idNumbers.begin(), idNumbers.end(), erridNumbersDB[i]));
    features.erase(features.begin() + index);
    libNames.erase(libNames.begin() + index);
    urls.erase(urls.begin() + index);
    idNumbers.erase(idNumbers.begin() + index);
  }

  // 此时的ids与features一一对应，更新人脸库
  assert(ids.size() == features.size() && ids.size() == libNames.size() &&
         ids.size() == urls.size() && ids.size() == idNumbers.size());

  // 备份图片
  for (size_t i = 0; i < ids.size(); ++i) {
    backupImage(ids[i], libNames[i], idNumbers[i], urls[i]);
  }

  // 归类不同的libName，libName和id和feature是一对多的关系
  std::unordered_map<std::string, std::pair<std::vector<long>,
                                            std::vector<std::vector<float>>>>
      libName2ids;
  for (size_t i = 0; i < ids.size(); ++i) {
    libName2ids[libNames[i]].first.emplace_back(ids[i]);
    libName2ids[libNames[i]].second.emplace_back(features[i]);
  }

  // 更新人脸库
  for (auto &item : libName2ids) {
    // 如果人脸库在线，需要更新人脸库，否则不需要
    if (core::FaceLibraryManager::getInstance().CHECK_FACELIB_EXIST(
            item.first)) {
      // 人脸库存在，直接插入。
      core::FaceLibraryManager::getInstance().createBatch(
          item.first, item.second.first, item.second.second);
    }
  }

  // 生成状态消息
  handleBatchErrors(erridNumbersAlgo, erridNumbersDB, status);

  return status;
}

// 批量更新
oatpp::Object<StatusDto>
FaceService::updateBatch(oatpp::Vector<oatpp::Object<FaceDto>> const &users) {
  auto status = StatusDto::createShared();

  std::vector<std::string> idNumbers;
  std::vector<std::string> libNames;
  std::vector<std::string> urls;
  for (size_t i = 0; i < users->size(); ++i) {
    idNumbers.push_back(users[i]->userId);
    urls.push_back(users[i]->url);
    libNames.push_back(users[i]->libName);
  }

  // 特征向量集合
  std::vector<std::vector<float>> features(users->size());

  // 算法执行失败的idNumber
  std::vector<std::string> erridNumbersAlgo;

  // 用于存储提取成功的特征向量的base64
  std::vector<std::string> vecsBase64;

  // 批量提取特征向量
  batchInfer(urls, idNumbers, features, erridNumbersAlgo);

  // 从idNumbers中移除失败的id，顺带移除features中的失败特征向量
  for (size_t i = 0; i < erridNumbersAlgo.size(); ++i) {
    // idNumbers 和 features
    // 一一对应，要获取idNumbers中的索引，再移除features中的特征向量 获取索引
    auto index = std::distance(
        idNumbers.begin(),
        std::find(idNumbers.begin(), idNumbers.end(), erridNumbersAlgo[i]));
    idNumbers.erase(idNumbers.begin() + index);
    features.erase(features.begin() + index);
    libNames.erase(libNames.begin() + index);
    urls.erase(urls.begin() + index);
  }

  // 此时的idNumbers与features一一对应
  assert(idNumbers.size() == features.size() &&
         idNumbers.size() == libNames.size() &&
         idNumbers.size() == urls.size());

  // 全部失败的情况
  if (idNumbers.empty()) {
    status->status = "Service Unavailable";
    status->code = 503;
    status->message = "All users failed to update.";
    return status;
  }

  // 数据库更新失败的情况
  std::vector<std::string> erridNumbersDB;

  // 更新人脸库时成功的id
  std::vector<long> ids;

  // 更新数据库
  for (size_t i = 0; i < idNumbers.size(); ++i) {
    auto id = FaceDBOperator::getInstance().updateUserByIdNumber(
        idNumbers[i], libNames[i], feature2base64(features[i]), status);
    if (id < 0) {
      // 更新数据库时失败项
      erridNumbersDB.push_back(idNumbers[i]);
    } else {
      ids.push_back(id);
    }
  }

  // 更新数据库失败的情况需要同步给features
  for (size_t i = 0; i < erridNumbersDB.size(); ++i) {
    // 从idNumbers中的索引，再移除features中的特征向量
    auto index = std::distance(
        idNumbers.begin(),
        std::find(idNumbers.begin(), idNumbers.end(), erridNumbersDB[i]));
    features.erase(features.begin() + index);
    libNames.erase(libNames.begin() + index);
    urls.erase(urls.begin() + index);
    idNumbers.erase(idNumbers.begin() + index);
  }

  // 此时的ids与features一一对应，更新人脸库
  assert(ids.size() == features.size() && ids.size() == libNames.size() &&
         ids.size() == urls.size() && ids.size() == idNumbers.size());

  // 备份图片
  for (size_t i = 0; i < ids.size(); ++i) {
    backupImage(ids[i], libNames[i], idNumbers[i], urls[i]);
  }

  // TODO:批量更新“换库”操作太麻烦了，暂时不做

  // 归类不同的libName，libName和id和feature是一对多的关系
  std::unordered_map<std::string, std::pair<std::vector<long>,
                                            std::vector<std::vector<float>>>>
      libName2ids;
  for (size_t i = 0; i < ids.size(); ++i) {
    libName2ids[libNames[i]].first.emplace_back(ids[i]);
    libName2ids[libNames[i]].second.emplace_back(features[i]);
  }

  // 更新人脸库
  for (auto &item : libName2ids) {
    // 检查人脸库是否在线，如果在线，直接更新，否则不需要
    if (core::FaceLibraryManager::getInstance().CHECK_FACELIB_EXIST(
            item.first)) {
      // 人脸库存在，直接更新
      core::FaceLibraryManager::getInstance().updateBatch(
          item.first, item.second.first, item.second.second);
    }
  }
  // 更新status
  handleBatchErrors(erridNumbersAlgo, erridNumbersDB, status);
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

  // 操作失败的idNumber
  std::vector<std::string> errIdNumbers;

  // 利用IdNumbers查询id
  std::vector<long> ids;
  for (size_t i = 0; i < IdNumbers.size(); ++i) {
    auto id =
        FaceDBOperator::getInstance().getIdByIdNumber(IdNumbers[i], status);
    if (id < 0) {
      // 查询失败的id
      errIdNumbers.push_back(IdNumbers[i]);
    } else {
      ids.push_back(id);
    }
  }

  // 如果没有有效的 ID，则直接返回
  if (ids.empty()) {
    status->status = "OK";
    status->code = 200;
    status->message = "No users to delete.";
    return status;
  }

  // 归类libName，分别批量删除
  std::unordered_map<std::string, std::vector<long>> libName2ids;
  for (size_t i = 0; i < ids.size(); ++i) {
    auto libName = FaceDBOperator::getInstance().getLibNameById(ids[i], status);
    if (!libName->empty()) {
      libName2ids[libName].emplace_back(ids[i]);
    }
  }

  // 构建 IN 子句中的参数字符串
  std::string idsString = "";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) {
      idsString += ", ";
    }
    idsString += std::to_string(ids[i]); // 确保转换为字符串
  }

  // 构建完整的 SQL 语句
  std::string sql = "DELETE FROM AppUser WHERE id IN (" + idsString + ");";

  // 执行 SQL 语句
  // auto dbResult = m_database->executeQuery(sql, {});
  auto dbResult = FaceDBOperator::getInstance().executeSql(sql);
  OATPP_ASSERT_HTTP(dbResult->isSuccess(), Status::CODE_500,
                    dbResult->getErrorMessage());

  // 删除人脸库
  for (auto &item : libName2ids) {
    // 检查人脸库是否在线，如果在线，直接删除，否则不需要
    if (core::FaceLibraryManager::getInstance().CHECK_FACELIB_EXIST(
            item.first)) {
      // 人脸库存在，直接删除
      core::FaceLibraryManager::getInstance().deleteBatch(item.first,
                                                          item.second);
    }
  }

  status->status = "OK";
  status->code = 200;
  status->message = "Users were successfully deleted.";
  return status;
}

// 人脸质检
oatpp::Object<StatusDto> FaceService::faceQuality(oatpp::String const &url) {
  auto status = StatusDto::createShared();

  int quality = -1;
  // 质量检测
  auto ret = core::FaceQuality::getInstance().infer(url, quality);
  if (!ret || quality < 0) {
    status->status = "Internal Server Error";
    status->code = 500;
    status->message = "Face quality check failed.";
    return status;
  }

  status->status = "OK";
  status->code = 200;
  status->message = std::to_string(quality);
  return status;
}

// Post 人脸质检
oatpp::Object<StatusDto>
FaceService::faceQuality(oatpp::Object<ImageDto> const &image) {
  // Post 本身只是解决了参数传递的问题，具体的逻辑还是调用Get的函数
  return faceQuality(image->url);
}
} // namespace server::face
