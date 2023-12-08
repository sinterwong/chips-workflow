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
#include <cstddef>
#include <cstdint>
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
  bool registerFacelib(std::string const &tname,
                       std::vector<long> const &ids = {},
                       std::vector<std::vector<float>> const &features = {});

  bool unregisterFacelib(std::string const &tname);

  bool createOne(std::string const &tname, long id, float *vec);

  bool createBatch(std::string const &tname, std::vector<long> &ids,
                   float **vecs);

  bool createBatch(std::string const &tname, std::vector<long> const &ids,
                   std::vector<std::vector<float>> const &features);

  bool updateOne(std::string const &tname, long id, float *vec);

  bool updateBatch(std::string const &tname, std::vector<long> &ids,
                   float **vecs);

  bool updateBatch(std::string const &tname, std::vector<long> const &ids,
                   std::vector<std::vector<float>> const &features);

  bool deleteOne(std::string const &tname, long id);

  bool deleteBatch(std::string const &tname, std::vector<long> &ids);

  long match(std::string const &tname, float *vec, float threshold);

  bool CHECK_FACELIB_EXIST(std::string const &tname) {
    if (!isFacelibExist(tname)) {
      FLOWENGINE_LOGGER_WARN("Facelib {} not exists. Please register first.",
                             tname);
      return false;
    }
    return true;
  }

  // 当前在线人脸库数量
  inline size_t getFacelibNum() { return facelibs.size(); }

  void printLibrary(std::string const &tname) {
    facelibs.at(tname)->printVectors();
  }

private:
  // facelib是否已经存在
  inline bool isFacelibExist(std::string const &tname) {
    return facelibs.find(tname) != facelibs.end();
  }

private:
  FaceLibraryManager() {
    // facelib = std::make_unique<FaceLibrary>(FACELIB_DIM);

    // if (std::filesystem::exists(outputPath)) {
    //   FLOWENGINE_LOGGER_INFO("Found the facelib will be loaded.");
    //   facelib->loadFromFile(outputPath);
    // } else {
    //   std::filesystem::create_directories(
    //       outputPath.substr(0, outputPath.find_last_of("/")));
    // }
  }
  ~FaceLibraryManager() {
    // facelib->saveToFile(outputPath);
    delete instance;
    instance = nullptr;
  }
  static FaceLibraryManager *instance;

private:
  // TODO:人脸库映射表用来分治人脸库，需要管理人脸库同时在线的数量
  std::unordered_map<std::string, std::unique_ptr<FaceLibrary>> facelibs;
};
} // namespace server::face::core
#endif
