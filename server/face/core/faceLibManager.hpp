/**
 * @file faceLibManager.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 人脸库，用于人脸特征的CRUD、定期备份等功能。全局单例
 * @version 0.1
 * @date 2023-10-24
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "facelib.hpp"
#include "logger/logger.hpp"
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

#ifndef __SERVER_FACE_CORE_FACE_LIBRARY_MANAGER_HPP_
#define __SERVER_FACE_CORE_FACE_LIBRARY_MANAGER_HPP_

namespace server::face::core {

class FaceLibraryManager {

public:
  static FaceLibraryManager &getInstance() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] { instance = new FaceLibraryManager(); });
    return *instance;
  }
  FaceLibraryManager(FaceLibraryManager const &) = delete;
  FaceLibraryManager &operator=(FaceLibraryManager const &) = delete;

public:
  bool createOne(long id, float *vec, bool isSave = true) {
    if (!facelib->addVector(vec, id)) {
      FLOWENGINE_LOGGER_WARN("Face library create {} failed.", id);
      return false;
    }

    if (isSave) {
      facelib->saveToFile(outputPath);
    }
    return true;
  }

  void createBatch(std::vector<long> &ids, float **vecs,
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

  bool updateOne(long id, float *vec, bool isSave = true) {
    if (!facelib->updateVector(id, vec)) {
      FLOWENGINE_LOGGER_WARN("Face library update {} failed.", id);
      return false;
    }
    if (isSave) {
      facelib->saveToFile(outputPath);
    }
    return true;
  }

  void updateBatch(std::vector<long> &ids, float **vecs,
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

  bool deleteOne(long id, bool isFave = true) {
    if (!facelib->deleteVector(id)) {
      FLOWENGINE_LOGGER_WARN("Face library delete {} failed.", id);
      return false;
    }
    if (isFave) {
      facelib->saveToFile(outputPath);
    }
    return true;
  }

  void deleteBatch(std::vector<long> &ids, std::vector<long> &err_ids) {

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

  long match(float *vec, float threshold) {
    auto ret = facelib->search(vec, 1).at(0);
    // 阈值
    FLOWENGINE_LOGGER_DEBUG("ID: {}, Distance: {}", ret.first, ret.second);
    if (ret.second > threshold) {
      return ret.first;
    }
    return -1;
  }

  void printLibrary() { facelib->printVectors(); }

private:
  FaceLibraryManager() {
    facelib = std::make_unique<FaceLibrary>(512);

    if (std::filesystem::exists(outputPath)) {
      FLOWENGINE_LOGGER_INFO("Found the facelib will be loaded.");
      facelib->loadFromFile(outputPath);
    } else {
      std::filesystem::create_directories(
          outputPath.substr(0, outputPath.find_last_of("/")));
    }
  }
  ~FaceLibraryManager() {
    facelib->saveToFile(outputPath);
    delete instance;
    instance = nullptr;
  }
  static FaceLibraryManager *instance;

private:
  // TODO:人脸库映射表用来分治人脸库，目前只管理了一个人脸库
  std::unordered_map<std::string, std::unique_ptr<FaceLibrary>> facelibs;

  std::string outputPath = "/public/face/facelib.bin";
  std::unique_ptr<FaceLibrary> facelib;
};
FaceLibraryManager *FaceLibraryManager::instance = nullptr;
} // namespace server::face::core
#endif
