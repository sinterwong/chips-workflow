/**
 * @file routeFramePool.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.2
 * @date 2022-06-30
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "routeFramePool.h"
#include <shared_mutex>
#include <utility>

RouteFramePool::RouteFramePool(int maxSize) {
  size = maxSize;
  for (int i = 0; i < size; i++) {
    FrameBuf temp;
    routeArray.emplace_back(temp);
  }
  key = 0;
}

FrameBuf RouteFramePool::read(int clock) {
  std::shared_lock<std::shared_mutex> lk(routeMutex);
  return routeArray[clock];
}

int RouteFramePool::write(FrameBuf buf) {
  std::lock_guard<std::shared_mutex> lk(routeMutex);
  routeArray[key].del();
  routeArray[key] = buf;
  int returnkey = key;
  key = key + 1 >= size ? 0 : key + 1;
  return returnkey;
}

RouteFramePool::~RouteFramePool() {
  std::lock_guard<std::shared_mutex> lk(routeMutex);
  for (auto &buf : routeArray) {
    buf.del();
  }
}

void RouteFramePool::checkSize() {
  std::lock_guard<std::shared_mutex> lk(routeMutex);
  routeArray[key].del();
}

void FrameBuf::del() {
  if (not isDel) {
    delFunction(dataList);
  }
  isDel = true;
}

std::any FrameBuf::read(std::string const& fname) {
  auto iter = mapFunction.find(fname);
  assert(iter != mapFunction.end());
  return iter->second(dataList, this);
}

void FrameBuf::write(std::vector<std::any> data, GFMap mFunc,
                     delFrameBufFunc dFunc, FrameInfo infor) {
  dataList = std::move(data);
  mapFunction = std::move(mFunc);
  delFunction = std::move(dFunc);
  std::tie(width, height, channel, type) = infor;
  isDel = false;
}
