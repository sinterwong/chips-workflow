/**
 * @file frameBuf.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-08-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <any>
#include <cassert>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef _FLOWCORE_FRAMEBUF_H_
#define _FLOWCORE_FRAMEBUF_H_
class FrameBuf;

// 获取帧的函数
using getFrameBufFunc =
    std::function<std::any(std::vector<std::any> &, FrameBuf *)>;

using delFrameBufFunc = std::function<void(std::vector<std::any> &)>; // 析构帧

using GFMap = std::unordered_map<std::string, getFrameBufFunc>; // 获取帧数据的

class FrameBuf {
protected:
  std::unordered_map<std::string, getFrameBufFunc> mapFunction;
  delFrameBufFunc delFunction;
  std::vector<std::any> dataList;

public:
  std::any read(std::string const &fname) {
    auto iter = mapFunction.find(fname);
    assert(iter != mapFunction.end());
    return iter->second(dataList, this);
  }

  void write(std::vector<std::any> data, GFMap mFunc, delFrameBufFunc dFunc) {
    dataList = std::move(data);
    mapFunction = std::move(mFunc);
    delFunction = std::move(dFunc);
  }

  ~FrameBuf() { delFunction(dataList); }
};

#endif