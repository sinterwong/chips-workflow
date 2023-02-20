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
#include <memory>
#include <unordered_map>

#ifndef __FLOWCORE_ALGORITHM_MANAGER_H_
#define __FLOWCORE_ALGORITHM_MANAGER_H_

namespace infer {
using algo_ptr = std::shared_ptr<AlgoInfer>;
class AlgorithmManager : public AlgorithmBus {

public:
  virtual ~AlgorithmManager() {}

  virtual bool registered(std::string const &name,
                          AlgorithmConfig const &) override;

  virtual bool unregistered(std::string const &name) override;

  virtual bool infer(std::string const &name, InferParams const &,
                     InferResult &ret) override;

protected:
  // 名称索引算法
  std::unordered_map<std::string, algo_ptr> name2algo;
};
#endif
}