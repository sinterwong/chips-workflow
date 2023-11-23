/**
 * @file faceLibManager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-10
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "faceLibManager.hpp"
#include <vector>

namespace server::face::core {

std::vector<float>
Flatten(std::vector<std::vector<float>> const &nestedVector) {
  size_t totalSize = 0;
  for (const auto &subVector : nestedVector) {
    totalSize += subVector.size();
  }

  std::vector<float> flatVector;
  flatVector.reserve(totalSize);

  for (const auto &subVector : nestedVector) {
    flatVector.insert(flatVector.end(), subVector.begin(), subVector.end());
  }
  // 这里的返回值编译器会有RVO/NRVO优化，不会有额外的拷贝开销
  return flatVector;
}

FaceLibraryManager *FaceLibraryManager::instance = nullptr;
bool FaceLibraryManager::registerFacelib(
    std::string const &name, std::vector<long> const &ids,
    std::vector<std::vector<float>> const &features) {
  if (facelibs.find(name) != facelibs.end()) {
    FLOWENGINE_LOGGER_WARN("Facelib {} already exists.", name);
    return false;
  }
  // 初始化人脸库
  facelibs[name] = std::make_unique<FaceLibrary>(FACELIB_DIM);
  if (!ids.empty() && !features.empty() && ids.size() == features.size()) {
    // 加载特征到人脸库
    try {
      // 此处的创建要么成功要么异常退出
      createBatch(name, ids, features);
    } catch (std::exception &e) {
      FLOWENGINE_LOGGER_ERROR("Create facelib {} failed: {}", name, e.what());
      facelibs.erase(name);
      return false;
    }
  }
  return true;
}

bool FaceLibraryManager::unregisterFacelib(std::string const &name) {
  if (!CHECK_FACELIB_EXIST(name)) { // 检查人脸库是否存在
    return false;
  }
  facelibs.erase(name);
  return true;
}

bool FaceLibraryManager::createOne(std::string const &tname, long id,
                                   float *vec) {

  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return false;
  }

  if (!facelibs.at(tname)->addVector(vec, id)) {
    FLOWENGINE_LOGGER_WARN("Face library create {} failed.", id);
    return false;
  }
  return true;
}

bool FaceLibraryManager::createBatch(std::string const &tname,
                                     std::vector<long> &ids, float **vecs) {
  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return false;
  }
  facelibs.at(tname)->addVectors(*vecs, ids);
  return true;
}

bool FaceLibraryManager::createBatch(
    std::string const &tname, std::vector<long> const &ids,
    std::vector<std::vector<float>> const &features) {
  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return false;
  }
  // 加载特征到人脸库
  auto ret = Flatten(features);
  facelibs.at(tname)->addVectors(ret.data(), ids);
  return true;
}

bool FaceLibraryManager::updateOne(std::string const &tname, long id,
                                   float *vec) {
  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return false;
  }
  if (!facelibs.at(tname)->updateVector(id, vec)) {
    FLOWENGINE_LOGGER_WARN("Face library update {} failed.", id);
    return false;
  }
  return true;
}

bool FaceLibraryManager::updateBatch(std::string const &tname,
                                     std::vector<long> &ids, float **vecs) {
  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return false;
  }
  facelibs.at(tname)->updateVectors(*vecs, ids);
  return true;
}

bool FaceLibraryManager::updateBatch(
    std::string const &tname, std::vector<long> const &ids,
    std::vector<std::vector<float>> const &features) {
  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return false;
  }
  // 加载特征到人脸库
  auto ret = Flatten(features);
  facelibs.at(tname)->updateVectors(ret.data(), ids);
  return true;
}

bool FaceLibraryManager::deleteOne(std::string const &tname, long id) {
  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return false;
  }
  if (!facelibs.at(tname)->deleteVector(id)) {
    FLOWENGINE_LOGGER_WARN("Face library delete {} failed.", id);
    return false;
  }
  return true;
}

bool FaceLibraryManager::deleteBatch(std::string const &tname,
                                     std::vector<long> &ids) {
  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return false;
  }
  facelibs.at(tname)->deleteVectors(ids);
  return true;
}

long FaceLibraryManager::match(std::string const &tname, float *vec,
                               float threshold) {
  if (!CHECK_FACELIB_EXIST(tname)) { // 检查人脸库是否存在
    return -2;
  }
  auto ret = facelibs.at(tname)->search(vec, 1).at(0);
  // 阈值
  FLOWENGINE_LOGGER_DEBUG("ID: {}, Distance: {}", ret.first, ret.second);
  if (ret.second > threshold) {
    return ret.first;
  }
  return -1;
}
} // namespace server::face::core