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
#include "visionInfer.hpp"
#include "logger/logger.hpp"
#include <memory>
#include <utility>

namespace infer {
bool AlgorithmManager::registered(std::string const &name, AlgorithmConfig const &config) { 
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

bool AlgorithmManager::unregistered(std::string const &name) { return true; }

bool AlgorithmManager::infer(std::string const &name, InferParams const &,
                             InferResult &ret) {
  return true;
}
}