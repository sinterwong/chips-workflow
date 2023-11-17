/**
 * @file test_facelib.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 人脸库管理功能测试
 * @version 0.1
 * @date 2023-10-30
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "algoManager.hpp"
#include "faceLibManager.hpp"
#include "logger/logger.hpp"

using namespace server::face;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

int main(int argc, char **argv) {

  FlowEngineLoggerSetLevel(1);

  bool ok;
  // 1. 新增人脸
  std::vector<float> feature1;
  auto r = core::AlgoManager::getInstance().infer("xxx.jpg", feature1);

  if (!r.get()) {
    FLOWENGINE_LOGGER_ERROR("Feature extract was failed.");
  }

  // 提取特征成功，接下来特征入库
  ok = core::FaceLibraryManager::getInstance().createOne("temp", 1,
                                                         feature1.data());

  if (!ok) {
    FLOWENGINE_LOGGER_ERROR("User failed to enter database.");
  } else {
    FLOWENGINE_LOGGER_INFO("User was successfully created.");
    core::FaceLibraryManager::getInstance().printLibrary("temp");
  }

  // 2. 检索人脸
  std::vector<float> feature2;
  core::AlgoManager::getInstance().infer("xxx.jpg", feature2);
  // 提取特征成功，接下来特征入库
  long idx = core::FaceLibraryManager::getInstance().match(
      "temp", feature2.data(), 0.5);
  FLOWENGINE_LOGGER_INFO("Id is {}.", idx);

  // 3. 更新人脸
  std::vector<float> feature3;
  core::AlgoManager::getInstance().infer("xxx.jpg", feature3);
  // 提取特征成功，接下来特征入库
  ok = core::FaceLibraryManager::getInstance().updateOne("temp", 1,
                                                         feature3.data());

  if (!ok) {
    FLOWENGINE_LOGGER_ERROR("User failed to update database.");
  } else {
    FLOWENGINE_LOGGER_INFO("User was successfully updated.");
    core::FaceLibraryManager::getInstance().printLibrary("temp");
  }

  // 4. 删除人脸
  ok = core::FaceLibraryManager::getInstance().deleteOne("temp", 1);
  if (!ok) {
    FLOWENGINE_LOGGER_ERROR("User failed to delete from database.");
  } else {
    FLOWENGINE_LOGGER_INFO("User was successfully deleted.");
    core::FaceLibraryManager::getInstance().printLibrary("temp");
  }

  return 0;
}