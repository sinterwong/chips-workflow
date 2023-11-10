/**
 * @file FaceService.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "FaceService.hpp"

namespace server::face {
constexpr float THRESHOLD = 0.33;


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

oatpp::Object<StatusDto> FaceService::createUser(oatpp::Int32 const &id,
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
    status->message = "User failed to create.";
  } else {
    status->status = "OK";
    status->code = 200;
    status->message = "User was successfully created.";
  }

  return status;
}

oatpp::Object<StatusDto> FaceService::updateUser(oatpp::Int32 const &id,
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

oatpp::Object<StatusDto> FaceService::deleteUser(oatpp::Int32 const &id) {

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

oatpp::Object<StatusDto> FaceService::searchUser(oatpp::String const &url) {
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
} // namespace server::face