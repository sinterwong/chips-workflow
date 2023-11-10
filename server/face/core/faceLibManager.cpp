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

namespace server::face::core {

FaceLibraryManager *FaceLibraryManager::instance = nullptr;
bool FaceLibraryManager::registerFacelib(std::string name, std::string path) {
  if (facelibs.find(name) != facelibs.end()) {
    FLOWENGINE_LOGGER_WARN("Facelib {} already exists.", name);
    return false;
  }
  facelibs[name] = std::make_unique<FaceLibrary>(FACELIB_DIM);
  return true;
}

bool FaceLibraryManager::unregisterFacelib(std::string name) {
  if (facelibs.find(name) == facelibs.end()) {
    FLOWENGINE_LOGGER_WARN("Facelib {} not exists.", name);
    return false;
  }
  facelibs.erase(name);
  return true;
}

bool FaceLibraryManager::createOne(long id, float *vec, bool isSave) {
  if (!facelib->addVector(vec, id)) {
    FLOWENGINE_LOGGER_WARN("Face library create {} failed.", id);
    return false;
  }

  if (isSave) {
    facelib->saveToFile(outputPath);
  }
  return true;
}

void FaceLibraryManager::createBatch(std::vector<long> &ids, float **vecs,
                                     std::vector<long> &err_ids) {

  // TODO:暂时先调用createOne，这样比较省事。后续开发真正的批量新增
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!createOne(ids.at(i), vecs[i], false)) {
      err_ids.push_back(ids.at(i));
    }
  }
  if (ids.size() > err_ids.size()) {
    // 存在成功更新的内容
    facelib->saveToFile(outputPath);
  }
}

bool FaceLibraryManager::updateOne(long id, float *vec, bool isSave) {
  if (!facelib->updateVector(id, vec)) {
    FLOWENGINE_LOGGER_WARN("Face library update {} failed.", id);
    return false;
  }
  if (isSave) {
    facelib->saveToFile(outputPath);
  }
  return true;
}

void FaceLibraryManager::updateBatch(std::vector<long> &ids, float **vecs,
                                     std::vector<long> &err_ids) {

  // TODO:暂时先调用updateOne，这样比较省事。后续开发真正的批量操作
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!updateOne(ids.at(i), vecs[i], false)) {
      err_ids.push_back(ids.at(i));
    }
  }
  if (ids.size() > err_ids.size()) {
    // 存在成功更新的内容
    facelib->saveToFile(outputPath);
  }
}

bool FaceLibraryManager::deleteOne(long id, bool isFave) {
  if (!facelib->deleteVector(id)) {
    FLOWENGINE_LOGGER_WARN("Face library delete {} failed.", id);
    return false;
  }
  if (isFave) {
    facelib->saveToFile(outputPath);
  }
  return true;
}

void FaceLibraryManager::deleteBatch(std::vector<long> &ids,
                                     std::vector<long> &err_ids) {

  // TODO:暂时先调用deleteOne，这样比较省事。后续开发真正的批量操作
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!deleteOne(ids.at(i), false)) {
      err_ids.push_back(ids.at(i));
    }
  }
  if (ids.size() > err_ids.size()) {
    // 存在成功更新的内容
    facelib->saveToFile(outputPath);
  }
}

long FaceLibraryManager::match(float *vec, float threshold) {
  auto ret = facelib->search(vec, 1).at(0);
  // 阈值
  FLOWENGINE_LOGGER_DEBUG("ID: {}, Distance: {}", ret.first, ret.second);
  if (ret.second > threshold) {
    return ret.first;
  }
  return -1;
}
} // namespace server::face::core