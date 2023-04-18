/**
 * @file routeFramePool.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.2
 * @date 2022-07-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef METAENGINE_ROUTEFRAMEPOOL_H
#define METAENGINE_ROUTEFRAMEPOOL_H

#include <any>
#include <atomic>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <shared_mutex>
#include <vector>

enum frameDataType {
  FLOAT32,
  FLOAT16,
  UINT8,
};

class FrameBuf;

using getFrameBufFunc = std::function<std::any(std::vector<std::any> &,
                                               FrameBuf *)>; // 获取帧的函数

using delFrameBufFunc = std::function<void(std::vector<std::any> &)>; // 析构帧

using FrameInfo = std::tuple<int, int, int, frameDataType>; // 数据帧的信息

using GFMap = std::map<std::string, getFrameBufFunc>; // 获取帧数据的

class FrameBuf {
protected:
  std::map<std::string, getFrameBufFunc> mapFunction;
  delFrameBufFunc delFunction;
  std::vector<std::any> dataList;

public:
  int width, height, channel;
  size_t size;
  frameDataType type;
  bool isDel = true;

  std::any read(std::string const &);

  void write(std::vector<std::any>, GFMap, delFrameBufFunc, FrameInfo);

  void del();
};

class FramePool {
public:
  virtual FrameBuf read(int) = 0;

  virtual int write(FrameBuf) = 0;

  virtual void checkSize() = 0;
};

class RouteFramePool : public FramePool {
protected:
  std::vector<FrameBuf> routeArray;
  std::shared_mutex routeMutex;
  int size, key;

public:
  RouteFramePool(int maxSize = 2);

  ~RouteFramePool();

  FrameBuf read(int) override;

  int write(FrameBuf) override;

  void checkSize() override;
};

#endif // METAENGINE_ROUTEFRAMEPOOL_H
