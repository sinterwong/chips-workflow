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
#include "logger/logger.hpp"
#include "visionInfer.hpp"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <utility>

namespace infer {
bool AlgorithmManager::registered(std::string const &name,
                                  AlgorithmConfig const &config) {
  std::lock_guard lk(m);
  auto iter = name2algo.find(name);
  if (iter != name2algo.end()) {
    FLOWENGINE_LOGGER_WARN("{} had registered!", name);
  }
  // 注册算法 TODO 先直接给到 VisionInfer
  algo_ptr algo = std::make_shared<VisionInfer>(config);
  if (!algo->init()) {
    FLOWENGINE_LOGGER_ERROR("algorithm manager: failed to register {}", name);
    return false;
  }
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
  return true;
}

bool AlgorithmManager::infer(std::string const &name, void *data,
                             InferParams const &params, InferResult &ret) {
  // infer调用的线程安全问题交给algoInfer去处理，可以做的更精细（如提前数据处理等）
  std::shared_lock lk(m);
  auto iter = name2algo.find(name);
  if (iter != name2algo.end()) {
    FLOWENGINE_LOGGER_ERROR("{} was no registered!", name);
    return false;
  }
  return iter->second->infer(data, params, ret);
}
} // namespace infer