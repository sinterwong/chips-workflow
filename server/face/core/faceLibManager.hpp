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
#include <filesystem>
#include <memory>

#ifndef __SERVER_FACE_CORE_FACE_LIBRARY_MANAGER_HPP_
#define __SERVER_FACE_CORE_FACE_LIBRARY_MANAGER_HPP_

namespace server::face::core {
using infer::solution::FaceLibrary;

class FaceLibraryManager {

public:
  static FaceLibraryManager &getInstance() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] { instance.reset(new FaceLibraryManager()); });
    return *instance;
  }
  FaceLibraryManager(FaceLibraryManager const &) = delete;
  FaceLibraryManager &operator=(FaceLibraryManager const &) = delete;

public:
  bool createOne(long id, float *vec) {
    facelib->addVector(vec, id);
    return true;
  }

  bool updateOne(long id, float *vec) {
    facelib->updateVector(id, vec);
    return true;
  }

  bool deleteOne(long id) {
    facelib->deleteVector(id);
    return true;
  }

  long match(float *vec, float threshold) {
    auto ret = facelib->search(vec, 1).at(0);
    // 阈值
    if (ret.second > threshold) {
      return ret.first;
    }
    return -1;
  }

private:
  FaceLibraryManager() {
    if (std::filesystem::exists(outputPath)) {
      facelib->loadFromFile(outputPath);
    } else {
      std::filesystem::create_directories(
          outputPath.substr(0, outputPath.find_last_of("/")));
      facelib = std::make_unique<FaceLibrary>(128);
    }
  }
  ~FaceLibraryManager() { facelib->saveToFile(outputPath); }
  static std::unique_ptr<FaceLibraryManager> instance;

private:
  std::string outputPath = "/public/face/feacelib.bin";
  std::unique_ptr<FaceLibrary> facelib;
};
std::unique_ptr<FaceLibraryManager> FaceLibraryManager::instance = nullptr;
} // namespace server::face::core
#endif