/**
 * @file streamManager.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 视频流管理器，用于注册、注销和生成视频帧数据。全局单例模式，VideoService中调用
 * @version 0.1
 * @date 2023-10-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __SERVER_FACE_CORE_STREAM_MANAGER_HPP_
#define __SERVER_FACE_CORE_STREAM_MANAGER_HPP_

#include "resultProcessor.hpp"
#include "thread_pool.h"
#include "video/videoDecode.hpp"
#include <cstdint>
#include <future>
#include <memory>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

using video::VideoDecode;

using video_ptr = std::unique_ptr<VideoDecode>;

using namespace std::chrono_literals;

namespace server::face::core {

class StreamManager {
public:
  // 获取StreamManager的唯一实例
  static StreamManager &getInstance() {
    static std::once_flag onceFlag; // 确保安全的初始化
    std::call_once(onceFlag, [] { instance = new StreamManager(); });
    return *instance;
  }

  // 禁止拷贝构造函数和拷贝赋值操作符
  StreamManager(StreamManager const &) = delete;
  StreamManager &operator=(StreamManager const &) = delete;

  bool registered(std::string const &name, std::string const &lname,
                  std::string const &uri);

  bool unregistered(std::string const &name);

private:
  StreamManager() {
    tpool = std::make_unique<thread_pool>();
    tpool->start(DECODER_POOL_SIZE);
  }

  ~StreamManager() {
    delete instance;
    instance = nullptr;
  };

  void checkCompletedFutures();

private:
  static StreamManager *instance;

  // 初始化解码线程池
  std::unique_ptr<thread_pool> tpool;
  std::vector<std::future<void>> futures;

  // 流对应表
  std::unordered_map<std::string, video_ptr> name2stream;

  std::shared_mutex m;
};
} // namespace server::face::core

#endif