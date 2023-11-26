/**
 * @file algoManager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-10
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "algoManager.hpp"

namespace server::face::core {
AlgoManager *AlgoManager::instance = nullptr;

std::future<bool> AlgoManager::infer(AlgoType const &atype,
                                     FrameInfo &frameInfo,
                                     InferResult &algoRet) {
  auto task = std::packaged_task<bool()>(
      [&] { return algoDispatchers.at(atype)->infer(frameInfo, algoRet); });

  std::future<bool> ret = task.get_future();

  // 在新线程上运行任务
  std::thread task_thread(std::move(task));
  task_thread.detach(); // 使线程在后台运行，独立于当前线程

  return ret; // 移交调用者等待算法结果
}

} // namespace server::face::core