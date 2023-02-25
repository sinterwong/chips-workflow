/**
 * @file algoInfer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 线程安全的算法推理接口，用于包装硬件的推理与算法的后处理
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "common/common.hpp"
#include "common/config.hpp"

#ifndef __FLOWCORE_ALGO_INFER_H_
#define __FLOWCORE_ALGO_INFER_H_

using common::AlgorithmConfig;
using common::InferParams;
using common::InferResult;

class AlgoInfer {
public:
  AlgoInfer(AlgorithmConfig const &config_) : config(config_) {}
  /**
   * @brief 初始化算法
   *
   * @return true
   * @return false
   */
  virtual bool init() = 0;

  /**
   * @brief 推理算法
   *
   * @param ret
   * @return true
   * @return false
   */
  virtual bool infer(void *data, InferParams const &, InferResult &ret) = 0;

  /**
   * @brief 关闭算法
   *
   * @return true
   * @return false
   */
  virtual bool destory() = 0;

protected:
  AlgorithmConfig config;
};
#endif