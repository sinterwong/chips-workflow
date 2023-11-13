/**
 * @file streamManager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "streamManager.hpp"

namespace server::face::core {
StreamManager *StreamManager::instance = nullptr;

bool StreamManager::registered(std::string const &name,
                               std::string const &uri) {
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

bool StreamManager::unregistered(std::string const &name) {
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

void StreamManager::checkCompletedFutures() {
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
} // namespace server::face::core