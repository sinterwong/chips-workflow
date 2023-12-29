/**
 * @file algoDispatcher.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __SERVER_FACE_CORE_ALGO_DISPATCHER_HPP_
#define __SERVER_FACE_CORE_ALGO_DISPATCHER_HPP_
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>

#include "logger/logger.hpp"
#include "preprocess.hpp"
#include "visionInfer.hpp"

namespace server::face::core {

// TODO:加载配置文件，从配置文件中获取相关算法的配置

static std::string DET_MODEL_PATH =
    "/opt/deploy/models/yolov8n_face_640x640.engine";

static std::string REC_MODEL_PATH = "/opt/deploy/models/arcface_112x112.engine";

static std::string QUALITY_MODEL_PATH =
    "/opt/deploy/models/face_quality_128x128.engine";

static std::string KEYPOINT_MODEL_PATH =
    "/opt/deploy/models/2d106det_192x192.engine";

enum class AlgoType { DET, REC, QUALITY, KEYPOINT };

using namespace infer;
using namespace common;

using algo_ptr = std::shared_ptr<AlgoInfer>;

class AlgoDispatcher {
private:
  std::vector<algo_ptr> algos;
  std::queue<algo_ptr> availableAlgos;

  std::mutex m;
  std::condition_variable cv;

private:
  algo_ptr getVision(AlgoConfig &&config) {

    std::shared_ptr<AlgoInfer> vision = std::make_shared<VisionInfer>(config);
    if (!vision->init()) {
      FLOWENGINE_LOGGER_ERROR("Failed to init vision");
      std::exit(-1); // 强制中断
      return nullptr;
    }
    return vision;
  }

  algo_ptr getAvailableAlgo() {
    std::unique_lock<std::mutex> lock(m);
    cv.wait(lock, [&]() { return !availableAlgos.empty(); });
    algo_ptr algo = availableAlgos.front();
    availableAlgos.pop();
    return algo;
  }

  void releaseAlgo(algo_ptr algo) {
    std::lock_guard<std::mutex> lock(m);
    availableAlgos.push(algo);
    cv.notify_one();
  }

  // 检测算法配置获取
  AlgoConfig getDetConfig() {
    PointsDetAlgo faceDet_config{{
                                     "faceDet",
                                     1,
                                     {"images"},
                                     {"output0"},
                                     DET_MODEL_PATH,
                                     "Yolov8PDet",
                                     {640, 640, 3},
                                     false,
                                     255.0,
                                     0,
                                     0.3,
                                 },
                                 5,
                                 0.4};
    AlgoConfig fdet_config;
    fdet_config.setParams(faceDet_config);
    return fdet_config;
  }

  // 识别算法配置获取
  AlgoConfig getRecConfig() {
    FeatureAlgo faceNet_config{{
                                   "faceRec",
                                   1,
                                   {"input.1"},
                                   {"516"},
                                   REC_MODEL_PATH,
                                   "FaceNet",
                                   {112, 112, 3},
                                   false,
                                   127.5,
                                   1.0,
                                   0.3,
                               },
                               512};
    AlgoConfig frec_config;
    frec_config.setParams(faceNet_config);
    return frec_config;
  }

  // 质量算法配置获取
  AlgoConfig getQualityConfig() {
    ClassAlgo faceQuality_config{{
        "faceQuality",
        1,
        {"input"},
        {"output"},
        QUALITY_MODEL_PATH,
        "Softmax",
        {128, 128, 3},
        false,
        255.0,
        0,
        0.3,
    }};
    AlgoConfig fquality_config;
    fquality_config.setParams(faceQuality_config);
    return fquality_config;
  }

  // 关键点算法配置获取
  AlgoConfig getKeyPointsConfig() {
    PointsDetAlgo facePoints_config{{
                                        "facePoints",
                                        1,
                                        {"data"},
                                        {"fc1"},
                                        KEYPOINT_MODEL_PATH,
                                        "FaceKeyPoints",
                                        {192, 192, 3},
                                        false,
                                        1,
                                        0,
                                        0.3,
                                    },
                                    106,
                                    0.4};
    AlgoConfig fpoints_config;
    fpoints_config.setParams(facePoints_config);
    return fpoints_config;
  }

public:
  // constructor
  AlgoDispatcher(AlgoType const &type, int const num) {
    switch (type) {
    case AlgoType::DET:
      for (int i = 0; i < num; ++i) {
        // 初始化算法资源
        algos.push_back(getVision(getDetConfig()));
        availableAlgos.push(algos[i]);
      }
      break;
    case AlgoType::REC:
      for (int i = 0; i < num; ++i) {
        // 初始化算法资源
        algos.push_back(getVision(getRecConfig()));
        availableAlgos.push(algos[i]);
      }
      break;
    case AlgoType::QUALITY:
      for (int i = 0; i < num; ++i) {
        // 初始化算法资源
        algos.push_back(getVision(getQualityConfig()));
        availableAlgos.push(algos[i]);
      }
      break;
    case AlgoType::KEYPOINT:
      for (int i = 0; i < num; ++i) {
        // 初始化算法资源
        algos.push_back(getVision(getKeyPointsConfig()));
        availableAlgos.push(algos[i]);
      }
      break;
    default: {
      FLOWENGINE_LOGGER_ERROR("AlgoType not supported");
      std::exit(-1); // 强制中断
    }
    }
  }

  // 获取可用算法资源并进行推理
  bool infer(FrameInfo &frame, InferResult &ret) {
    InferParams params{std::string("xxx"),
                       frame.type,
                       0.0,
                       {"xxx"},
                       {frame.inputShape.at(0), frame.inputShape.at(1),
                        frame.inputShape.at(2)}};
    algo_ptr vision = getAvailableAlgo();
    // TODO:infer 是线程安全的，此举多余，后续优化一下策略
    bool res = vision->infer(frame, params, ret);
    releaseAlgo(vision);
    return res;
  }
};

} // namespace server::face::core

#endif // __SERVER_FACE_CORE_ALGO_DISPATCHER_HPP_