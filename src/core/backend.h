/**
 * @file backend.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.2
 * @date 2022-08-11
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef FLOWCORE_BACKEND_H
#define FLOWCORE_BACKEND_H

#include "algorithmBus.h"
#include "messageBus.h"
#include "streamPoolBus.hpp"
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Backend {
public:
  std::unique_ptr<MessageBus> message; // 管理注册，关闭和模块间的通讯
  std::unique_ptr<StreamPoolBus> pools; // 管理视频帧
  std::unique_ptr<AlgorithmBus> algo;  // 管理算法注册，关闭和推理

  Backend(std::unique_ptr<MessageBus> &&message_,
          std::unique_ptr<StreamPoolBus> &&pool_,
          std::unique_ptr<AlgorithmBus> &&algo_)
      : message(std::move(message_)), pools(std::move(pool_)),
        algo(std::move(algo_)) {}
};

#endif // FLOWCORE_BACKEND_H
