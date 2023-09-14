/**
 * @file algorithmManager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "algorithmManager.hpp"
#include "infer/preprocess.hpp"
#include "logger/logger.hpp"
#include "visionInfer.hpp"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace infer {
bool AlgorithmManager::registered(std::string const &name,
                                  AlgoConfig const &config) {
  std::lock_guard lk(m);
  auto iter = name2algo.find(name);
  if (iter != name2algo.end()) {
    FLOWENGINE_LOGGER_WARN("{} had registered!", name);
    return false;
  }
  // 注册算法 TODO 先直接给到 VisionInfer，话说这套框架有可能加入非视觉任务？
  name2algo[name] = std::make_shared<VisionInfer>(config);
  if (!name2algo.at(name)->init()) {
    FLOWENGINE_LOGGER_CRITICAL("algorithm manager: failed to register {}",
                               name);
    return false;
  }
  FLOWENGINE_LOGGER_INFO(
      "AlgorithmManager algorithm {} registered successfully", name);
  return true;
}

bool AlgorithmManager::unregistered(std::string const &name) {
  std::lock_guard lk(m);
  auto iter = name2algo.find(name);
  if (iter == name2algo.end()) {
    FLOWENGINE_LOGGER_WARN("{} was no registered!", name);
    return false;
  }
  // TODO 硬件infer的析构会关闭资源，但是线程不安全，应该在visionInfer部分去控制
  // iter = name2algo.erase(iter);
  if (!iter->second->destory()) {
    FLOWENGINE_LOGGER_ERROR(
        "AlgorithmManager unregistered: failed to destory algorithm {}", name);
    return false;
  }
  iter = name2algo.erase(iter);
  FLOWENGINE_LOGGER_INFO(
      "AlgorithmManager algorithm {} unregistered successfully", name);
  return true;
}

bool AlgorithmManager::infer(std::string const &name, void *data,
                             InferParams const &params, InferResult &ret) {

  Shape const &shape = params.shape;
  auto cv_type = shape.at(2) == 1 ? CV_8UC1 : CV_8UC3;
  cv::Mat image = cv::Mat(shape.at(1), shape.at(0), cv_type, data);
  auto &bbox = params.region;
  cv::Mat inferImage;
  cv::Rect2i rect{static_cast<int>(bbox.second[0]),
                  static_cast<int>(bbox.second[1]),
                  static_cast<int>(bbox.second[2] - bbox.second[0]),
                  static_cast<int>(bbox.second[3] - bbox.second[1])};
  if (rect.area() != 0) {
    if (!utils::cropImage(image, inferImage, rect, params.frameType,
                          params.cropScaling)) {
      FLOWENGINE_LOGGER_ERROR(
          "VisionInfer infer: cropImage is failed, rect is {},{},{},{}, "
          "but the video resolution is {}x{}. The error comes from {}-{}.",
          rect.x, rect.y, rect.width, rect.height, image.cols, image.rows,
          params.name, name);
      return false;
    }
  } else {
    inferImage = image.clone();
  }

  FrameInfo frame;
  switch (params.frameType) {
  case common::ColorType::None:
  case common::ColorType::RGB888:
  case common::ColorType::BGR888:
    frame.shape = {inferImage.cols, inferImage.rows, 3};
    break;
  case common::ColorType::NV12:
    frame.shape = {inferImage.cols, inferImage.rows * 2 / 3, 3};
    break;
  }
  frame.type = params.frameType;
  frame.data = reinterpret_cast<void **>(&inferImage.data);

  /* TODO
   * 共享锁饥饿问题（推理线程交替执行且耗时，导致注册和注销线程无法获取到锁）
   */
  std::shared_lock lk(m);
  auto iter = name2algo.find(name);
  if (iter == name2algo.end()) {
    FLOWENGINE_LOGGER_WARN("{} was no registered!", name);
    return false;
  }
  // infer调用的线程安全问题交给algoInfer去处理，可以做的更精细（如前后数据处理等）
  return iter->second->infer(frame, params, ret);
}
} // namespace infer