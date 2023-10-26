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

#include "thread_pool.h"
#include "video/videoDecode.hpp"
#include "resultProcessor.hpp"
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
    std::call_once(onceFlag, [] { instance.reset(new StreamManager()); });
    return *instance;
  }

  // 禁止拷贝构造函数和拷贝赋值操作符
  StreamManager(StreamManager const &) = delete;
  StreamManager &operator=(StreamManager const &) = delete;

  bool registered(std::string const &name, std::string const &uri) {
    // 检查已使用线程的数量
    checkCompletedFutures();
    if (futures.size() >= DECODER_POOL_SIZE) {
      // 超过了可启动数量
      FLOWENGINE_LOGGER_ERROR(
          "The number of bootable videos has reached the upper limit.");
      return false;
    }

    std::lock_guard lk(m);
    auto iter = name2stream.find(name);
    if (iter != name2stream.end()) {
      FLOWENGINE_LOGGER_WARN("{} had registered!", name);
      return false;
    }
    // 注册流
    name2stream[name] = std::make_unique<VideoDecode>(uri);

    // 调用线程启动输出帧的任务
    auto f = tpool->submit([this, name]() -> void {
      while (1) { // 不解除注册就无限重启
        if (!name2stream.at(name)->init()) {
          FLOWENGINE_LOGGER_ERROR("init decoder failed!");
          return;
        }
        FLOWENGINE_LOGGER_INFO("Video manager has initialized!");

        if (!name2stream.at(name)->run()) {
          FLOWENGINE_LOGGER_ERROR("run decoder failed!");
          return;
        }
        FLOWENGINE_LOGGER_INFO("Video manager is running!");
        uint32_t count = 0;
        while (name2stream.at(name)->isRunning()) {
          auto image = name2stream.at(name)->getcvImage();
          if (!image) {
            FLOWENGINE_LOGGER_ERROR("get image failed!");
            continue;
          }
          // 抽帧处理
          if (count++ % 5 == 0) {
            continue;
          }
          // 输出帧提供给结果处理器，结果处理器中负责任务的调度和结果分析
          // cv::imwrite("test_stream_manager.jpg", *image);
          FramePackage frame{name, image};
          ResultProcessor::getInstance().onFrameReceived(std::move(frame));
        }
        name2stream.at(name)->stop();
        std::this_thread::sleep_for(500ms);
      }
    });
    futures.emplace_back(std::move(f));
    return true;
  }

  bool unregistered(std::string const &name) {
    std::lock_guard lk(m);
    auto iter = name2stream.find(name);
    if (iter == name2stream.end()) {
      FLOWENGINE_LOGGER_WARN("{} was no registered!", name);
      return false;
    }
    // 内部释放足够安全，直接析构就可以
    iter = name2stream.erase(iter);
    return true;
  }

private:
  StreamManager() {
    tpool = std::make_unique<thread_pool>();
    tpool->start(DECODER_POOL_SIZE);
  }

  ~StreamManager() = default;

  void checkCompletedFutures() {
    std::lock_guard lock(m);
    auto it = futures.begin();
    while (it != futures.end()) {
      if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        // 如果future已经完成，从vector中移除
        it = futures.erase(it);
      } else {
        ++it;
      }
    }
  }

private:
  static std::unique_ptr<StreamManager> instance;

  // 初始化解码线程池
  std::unique_ptr<thread_pool> tpool;
  std::vector<std::future<void>> futures{DECODER_POOL_SIZE};

  // 流对应表
  std::unordered_map<std::string, video_ptr> name2stream;

  std::shared_mutex m;
};
std::unique_ptr<StreamManager> StreamManager::instance = nullptr;
} // namespace server::face::core

#endif