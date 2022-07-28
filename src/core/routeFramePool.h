//
// Created by Wallel on 2022/2/16.
//

#ifndef METAENGINE_ROUTEFRAMEPOOL_H
#define METAENGINE_ROUTEFRAMEPOOL_H

#include <any>
#include <atomic>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <vector>

// #include <boost/pool/pool.hpp>
// #include <boost/thread.hpp>
#include <shared_mutex>

enum frameDataType {
  FLOAT32,
  FLOAT16,
  UINT8,
};

class FrameBuf {
protected:
  std::map<std::string,
           std::function<std::any(std::vector<std::any> &, FrameBuf *)>>
      mapFunction;
  std::function<void(std::vector<std::any> &)> delFunction;
  std::vector<std::any> dataList;

public:
  int width, height, channel;
  size_t size;
  frameDataType type;
  bool isDel = true;

  std::any read(std::string str = "void*");

  void write(
      std::vector<std::any>,
      std::map<std::string,
               std::function<std::any(std::vector<std::any> &, FrameBuf *)>
               >,
      std::function<void(std::vector<std::any> &)>,
      std::tuple<int, int, int, frameDataType>);

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
  // boost::pool<> framePool = boost::pool<>(sizeof(char));
  std::vector<FrameBuf> routeArray;
  // boost::shared_mutex routeMutex;
  std::shared_mutex routeMutex;
  int size, key;

  int defaultHeight, defaultWidth, defaultChannel;

public:
  RouteFramePool(int maxSize = 2, int width = 1920, int height = 1080,
                 int channel = 3);

  // RouteFramePool(RouteFramePool &&other);

  // RouteFramePool(RouteFramePool &other);

  // RouteFramePool &operator=(RouteFramePool &&other);

  ~RouteFramePool();

  FrameBuf read(int) override;

  int write(FrameBuf) override;

  void checkSize() override;
};

#endif // METAENGINE_ROUTEFRAMEPOOL_H
