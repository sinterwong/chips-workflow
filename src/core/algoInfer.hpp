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

#ifndef __FLOWCORE_ALGO_INFER_H_
#define __FLOWCORE_ALGO_INFER_H_

using common::AlgoConfig;
using common::AlgoRetType;
using common::AlgoSerial;
using common::InferParams;
using common::InferResult;

class AlgoInfer {
public:
  AlgoInfer(AlgoConfig const &config_) : config(config_) {}
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
   * @brief 获取算法类型
   *
   */
  virtual AlgoRetType getType() const noexcept = 0;

  /**
   * @brief 获取算法系列
   *
   */
  virtual AlgoSerial getSerial() const noexcept = 0;

  /**
   * @brief 关闭算法
   *
   * @return true
   * @return false
   */
  virtual bool destory() noexcept = 0;

protected:
  AlgoConfig config;
};
#endif