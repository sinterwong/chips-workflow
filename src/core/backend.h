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

#include "messageBus.h"
#include "routeFramePool.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Backend {
public:
  std::unique_ptr<MessageBus> message;
  std::unique_ptr<RouteFramePool> pool;

Backend(std::unique_ptr<MessageBus> &&message_,
          std::unique_ptr<RouteFramePool> &&pool_)
      : message(std::move(message_)), pool(std::move(pool_)) {}

};

#endif // FLOWCORE_BACKEND_H
