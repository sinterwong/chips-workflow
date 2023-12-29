/**
 * @file resultProcessor.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 结果处理器，协调组件，从流管理器获取帧数据，调用算法管理器获取算法结果，调用人脸资源管理器匹
 * 配算法结果，根据匹配结果调用后端服务，全局单例
 * @version 0.1
 * @date 2023-10-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "common/thread_pool.h"
#include "faceLibManager.hpp"
#include "faceRecognition.hpp"
#include "logger/logger.hpp"
#include "networkUtils.hpp"
#include <cstddef>
#include <curl/curl.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <unordered_map>
#include <vector>

#ifndef __SERVER_FACE_CORE_RESULT_PROCESSOR_HPP_
#define __SERVER_FACE_CORE_RESULT_PROCESSOR_HPP_

namespace server::face::core {

struct PostInfo {
  std::string cameraName;
  std::string idNumber;
};

class ResultProcessor {
public:
  static ResultProcessor &getInstance() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] { instance = new ResultProcessor(); });
    return *instance;
  }
  ResultProcessor(ResultProcessor const &) = delete;
  ResultProcessor &operator=(ResultProcessor const &) = delete;

public:
  void onFrameReceived(std::string const &lname, FramePackage framePackage,
                       std::string const &postUrl);

private:
  ResultProcessor() {
    tpool = std::make_unique<thread_pool>();
    tpool->start(4);
  }
  ~ResultProcessor() {
    delete instance;
    instance = nullptr;
  }
  static ResultProcessor *instance;

private:
  // 执行一次算法任务
  std::unique_ptr<thread_pool> tpool;

private:
  void oneProcess(std::string const &lname, FramePackage framePackage,
                  std::string const &postUrl);
};
} // namespace server::face::core

#endif
