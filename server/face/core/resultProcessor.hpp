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

#include "algoManager.hpp"
#include "curlUtils.hpp"
#include "faceLibManager.hpp"
#include "faceRecognition.hpp"
#include "logger/logger.hpp"
#include "thread_pool.h"
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
  long id;
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
  void onFrameReceived(FramePackage &&framePackage) {
    FLOWENGINE_LOGGER_DEBUG("Frame received and processed.");
    tpool->submit(
        [this, &framePackage]() { this->oneProcess(std::move(framePackage)); });
  }

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

  // 后端请求接口
  std::string postUrl = "http://localhost:19797";

private:
  void oneProcess(FramePackage &&framePackage) {
    // 单个任务的函数
    // * 1. 根据帧包中的图片，送入人脸识别算法
    std::vector<float> feature;
    auto f = AlgoManager::getInstance().infer(framePackage, feature);
    if (!f.get()) {
      return;
    }

    // * 2. 人脸识别结果匹配人脸库
    auto idx = FaceLibraryManager::getInstance().match(feature.data(), 0.8);
    if (idx < 0) { // 没有匹配到人脸
      return;
    }

    // * 3. 根据人脸库返回的结果决定是否发送消息到后端服务
    PostInfo postInfo{framePackage.cameraName, idx};
    MY_CURL_POST(postUrl, {
      info["cameraName"] = postInfo.cameraName;
      info["id"] = postInfo.id;
    });
  }
};
ResultProcessor *ResultProcessor::instance = nullptr;
} // namespace server::face::core

#endif
