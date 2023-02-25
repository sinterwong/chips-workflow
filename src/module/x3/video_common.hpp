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
#include <mutex>

namespace module::utils {
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

  void setChannel(int i) {
    std::lock_guard<std::mutex> lk(m_);
    assert(i < size);
    channelsStatus[i].store(true);
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

} // namespace module::utils
#endif