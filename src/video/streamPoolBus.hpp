/**
 * @file streamPoolBus.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-08-11
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "common/common.hpp"
#include "framePool.hpp"
#include "logger/logger.hpp"
#include <memory>

#ifndef __FLOWCORE_STREAM_POOL_BUS_H_
#define __FLOWCORE_STREAM_POOL_BUS_H_

using pool_ptr = std::unique_ptr<RouteFramePool>;

class StreamPoolBus {
public:
  explicit StreamPoolBus() = default;

  ~StreamPoolBus() {}

  bool registered(std::string const &name, int maxSize = 2) {
    std::lock_guard lk(m);
    auto iter = name2pool.find(name);
    if (iter != name2pool.end()) {
      FLOWENGINE_LOGGER_WARN("{} had registered!", name);
      return false;
    }
    // 注册帧池
    name2pool[name] = std::make_unique<RouteFramePool>(maxSize);
    return true;
  }

  bool unregistered(std::string const &name) {
    std::lock_guard lk(m);
    auto iter = name2pool.find(name);
    if (iter == name2pool.end()) {
      FLOWENGINE_LOGGER_WARN("{} was no registered!", name);
      return false;
    }
    iter = name2pool.erase(iter);
    return true;
  }

  frame_ptr read(std::string const &name, int key) {
    std::shared_lock lk(m);
    auto iter = name2pool.find(name);
    if (iter == name2pool.end()) {
      FLOWENGINE_LOGGER_WARN("{} was no registered!", name);
      return nullptr;
    }
    return name2pool.at(name)->read(key);
  }

  int write(std::string const &name, frame_ptr buf) {
    std::shared_lock lk(m);
    auto iter = name2pool.find(name);
    if (iter == name2pool.end()) {
      FLOWENGINE_LOGGER_WARN("{} was no registered!", name);
      return -1;
    }
    int key = name2pool.at(name)->write(buf);
    return key;
  }

protected:
  // 名称索引算法
  std::unordered_map<std::string, pool_ptr> name2pool;
  std::shared_mutex m;
};
#endif