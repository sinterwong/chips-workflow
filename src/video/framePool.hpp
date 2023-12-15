/**
 * @file framePool.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-08-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef _FLOWCORE_FRAMEPOOL_H_
#define _FLOWCORE_FRAMEPOOL_H_

#include "frameBuf.hpp"
#include <any>
#include <atomic>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <shared_mutex>
#include <vector>

enum frameDataType {
  FLOAT32,
  FLOAT16,
  UINT8,
};

using frame_ptr = std::shared_ptr<FrameBuf>;

class FramePool {
public:
  virtual frame_ptr read(int) = 0;
  virtual int write(frame_ptr) = 0;
};

class RouteFramePool : public FramePool {
protected:
  std::vector<frame_ptr> routeArray;
  std::shared_mutex routeMutex;
  int size, key;

public:
  RouteFramePool(int maxSize = 2) {
    size = maxSize;
    routeArray.resize(size);
    key = 0;
  }

  ~RouteFramePool() {}

  frame_ptr read(int clock) override {
    std::shared_lock<std::shared_mutex> lk(routeMutex);
    return routeArray[clock];
  }

  int write(frame_ptr buf) override {
    std::lock_guard<std::shared_mutex> lk(routeMutex);
    routeArray.at(key) = buf;
    int returnkey = key;
    key = key + 1 >= size ? 0 : key + 1;
    return returnkey;
  }
};

#endif // METAENGINE_ROUTEFRAMEPOOL_H
