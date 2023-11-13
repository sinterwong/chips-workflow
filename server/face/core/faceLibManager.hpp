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
#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#ifndef __SERVER_FACE_CORE_FACE_LIBRARY_MANAGER_HPP_
#define __SERVER_FACE_CORE_FACE_LIBRARY_MANAGER_HPP_

#define FACELIB_DIM 512

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
  bool registerFacelib(std::string name, std::string path);

  bool unregisterFacelib(std::string name);

  // 从数据库中获取人脸特征，然后装载到人脸库中
  bool loadFacelib(std::vector<long> const &ids,
                   std::vector<std::vector<float>> const &features);

  bool createOne(long id, float *vec);

  void createBatch(std::vector<long> &ids, float **vecs);

  bool updateOne(long id, float *vec);

  void updateBatch(std::vector<long> &ids, float **vecs);

  bool deleteOne(long id);

  void deleteBatch(std::vector<long> &ids);

  long match(float *vec, float threshold);

  void printLibrary() { facelib->printVectors(); }

private:
  FaceLibraryManager() {
    facelib = std::make_unique<FaceLibrary>(FACELIB_DIM);

    // if (std::filesystem::exists(outputPath)) {
    //   FLOWENGINE_LOGGER_INFO("Found the facelib will be loaded.");
    //   facelib->loadFromFile(outputPath);
    // } else {
    //   std::filesystem::create_directories(
    //       outputPath.substr(0, outputPath.find_last_of("/")));
    // }
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
} // namespace server::face::core
#endif
