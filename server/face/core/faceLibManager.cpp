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

bool FaceLibraryManager::createOne(long id, float *vec) {
  if (!facelib->addVector(vec, id)) {
    FLOWENGINE_LOGGER_WARN("Face library create {} failed.", id);
    return false;
  }
  return true;
}

void FaceLibraryManager::createBatch(std::vector<long> &ids, float **vecs) {
  facelib->addVectors(*vecs, ids);
}

void FaceLibraryManager::createBatch(
    std::vector<long> const &ids,
    std::vector<std::vector<float>> const &features) {
  // 加载特征到人脸库
  auto ret = Flatten(features);
  facelib->addVectors(ret.data(), ids);
}

bool FaceLibraryManager::updateOne(long id, float *vec) {
  if (!facelib->updateVector(id, vec)) {
    FLOWENGINE_LOGGER_WARN("Face library update {} failed.", id);
    return false;
  }
  return true;
}

void FaceLibraryManager::updateBatch(std::vector<long> &ids, float **vecs) {
  facelib->updateVectors(*vecs, ids);
}

void FaceLibraryManager::updateBatch(
    std::vector<long> const &ids,
    std::vector<std::vector<float>> const &features) {
  // 加载特征到人脸库
  auto ret = Flatten(features);
  facelib->updateVectors(ret.data(), ids);
}

bool FaceLibraryManager::deleteOne(long id) {
  if (!facelib->deleteVector(id)) {
    FLOWENGINE_LOGGER_WARN("Face library delete {} failed.", id);
    return false;
  }
  return true;
}

void FaceLibraryManager::deleteBatch(std::vector<long> &ids) {
  facelib->deleteVectors(ids);
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