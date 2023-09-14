/**
 * @file algorithmManager.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "core/algoInfer.hpp"
#include "core/algorithmBus.h"
#include <algorithm>
#include <memory>
#include <shared_mutex>
#include <unordered_map>

#ifndef __FLOWCORE_ALGORITHM_MANAGER_H_
#define __FLOWCORE_ALGORITHM_MANAGER_H_

namespace infer {
using algo_ptr = std::shared_ptr<AlgoInfer>;
class AlgorithmManager : public AlgorithmBus {

public:
  virtual ~AlgorithmManager() {}

  virtual bool registered(std::string const &, AlgoConfig const &) override;

  virtual bool unregistered(std::string const &) override;

  virtual bool infer(std::string const &, void *, InferParams const &,
                     InferResult &) override;

  virtual AlgoRetType getType(std::string const &name) const noexcept override {
    return name2algo.at(name)->getType();
  };

  virtual AlgoSerial
  getSerial(std::string const &name) const noexcept override {
    return name2algo.at(name)->getSerial();
  };

  virtual void
  getActiveAlgorithms(std::vector<std::string> &activedAlgorithms) override {
    std::shared_lock<std::shared_mutex> lock(m); // 为读取加锁
    activedAlgorithms.reserve(name2algo.size()); // 预留空间，优化性能
    std::transform(name2algo.begin(), name2algo.end(),
                   std::back_inserter(activedAlgorithms),
                   [](const auto &pair) { return pair.first; });
  }

protected:
  // 名称索引算法
  std::unordered_map<std::string, algo_ptr> name2algo;
  std::shared_mutex m;
};
#endif
}