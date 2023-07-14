/**
 * @file video_common.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-12
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef _X3_VIDEO_COMMON_HPP_
#define _X3_VIDEO_COMMON_HPP_

#include "uri.hpp"
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <list>
#include <mutex>

using namespace video::utils;
namespace video {

/**
 * @brief
 * 队列的方式管理全局的channel数，0~31。
 * 使用队列管理不会导致刚刚释放的channel立即被再次使用，一定程度上避免编解码器释放之前该channel再次被使用
 */
class ChannelsQueueManager {
public:
  static ChannelsQueueManager &getInstance() {
    static ChannelsQueueManager instance;
    return instance;
  }

  int getChannel() {
    std::lock_guard<std::mutex> lk(m_);
    if (channelsStatus.empty()) {
      return -1;
    }
    int channel = channelsStatus.front();
    channelsStatus.pop_front();
    return channel;
  }

  bool setChannel(int i) {
    if (i < 0 || i >= size) {
      return false;
    }
    std::lock_guard<std::mutex> lk(m_);
    channelsStatus.push_back(i);
    return true;
  }

private:
  ChannelsQueueManager() {
    for (int i = 0; i < size; i++) {
      channelsStatus.push_back(i);
    }
  }
  ~ChannelsQueueManager() = default;
  ChannelsQueueManager(const ChannelsQueueManager &) = delete;
  ChannelsQueueManager &operator=(const ChannelsQueueManager &) = delete;

  std::mutex m_;
  std::list<int> channelsStatus;
  static constexpr int size = 32;
};

/**
 * @brief 提供全局的channel管理
 *
 */
class ChannelsManager {
public:
  static ChannelsManager &getInstance() {
    static ChannelsManager instance;
    return instance;
  }

  int getChannel() {
    std::lock_guard<std::mutex> lk(m_);
    for (int i = 0; i < size; i++) {
      if (channelsStatus[i].load()) {
        channelsStatus[i].store(false);
        return i;
      }
    }
    return -1;
  }

  bool setChannel(int i) {
    if (i < 0 || i >= size) {
      return false;
    }
    std::lock_guard<std::mutex> lk(m_);
    channelsStatus[i].store(true);
    return true;
  }

private:
  ChannelsManager() {
    for (auto &status : channelsStatus) {
      status.store(true);
    }
  };
  ChannelsManager(const ChannelsManager &) = delete;
  ChannelsManager &operator=(const ChannelsManager &) = delete;
  ChannelsManager(ChannelsManager &&) = delete;
  ChannelsManager &operator=(ChannelsManager &&) = delete;

  std::mutex m_;
  std::array<std::atomic_bool, 32> channelsStatus;
  const int size = channelsStatus.size();
};

/**
 * @brief 视频组件初始化时的选项
 *
 */
struct videoOptions {
  URI resource;
  int width;
  int height;
  int frameRate;
  int videoIdx;
};

} // namespace video
#endif