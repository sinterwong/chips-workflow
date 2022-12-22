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

RouteFramePool::RouteFramePool(int maxSize, int width, int height,
                               int channel) {
  size = maxSize;
  defaultHeight = height;
  defaultWidth = width;
  defaultChannel = channel;
  for (int i = 0; i < size; i++) {
    FrameBuf temp;
    routeArray.emplace_back(temp);
  }
  key = 0;
}

FrameBuf RouteFramePool::read(int clock) {
  std::shared_lock<std::shared_mutex> lk(routeMutex);
  // routeMutex.lock_shared();
  FrameBuf temp = routeArray[clock];
  // routeMutex.unlock_shared();
  return temp;
}

int RouteFramePool::write(FrameBuf buf) {
  std::lock_guard<std::shared_mutex> lk(routeMutex);
  // routeMutex.lock_upgrade();
  routeArray[key].del();
  routeArray[key] = buf;
  int returnkey = key;
  key = key + 1 >= size ? 0 : key + 1;
  // routeMutex.unlock_upgrade();
  return returnkey;
}

RouteFramePool::~RouteFramePool() {
  for (auto &buf : routeArray) {
    buf.del();
  }
}

void RouteFramePool::checkSize() { routeArray[key].del(); }

void FrameBuf::del() {
  if (not isDel) {
    delFunction(dataList);
  }
  isDel = true;
}

std::any FrameBuf::read(std::string str) {
  auto iter = mapFunction.find(str);
  assert(iter != mapFunction.end());
  return iter->second(dataList, this);
}

void FrameBuf::write(
    std::vector<std::any> data,
    std::map<std::string,
             std::function<std::any(std::vector<std::any> &, FrameBuf *)>>
        mFunc,
    std::function<void(std::vector<std::any> &)> dFunc,
    std::tuple<int, int, int, frameDataType> infor) {
  dataList = std::move(data);
  mapFunction = std::move(mFunc);
  delFunction = std::move(dFunc);
  std::tie(width, height, channel, type) = infor;
  isDel = false;
}
