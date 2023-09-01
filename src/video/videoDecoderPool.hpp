/**
 * @file videoDecoderPool.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-08-31
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "videoDecode.hpp"
#include <queue>

#ifndef __FLOWENGINE_VIDEO_DECODER_POOL_H_
#define __FLOWENGINE_VIDEO_DECODER_POOL_H_

#ifndef DECODER_POOL_SIZE
#define DECODER_POOL_SIZE 4
#endif

namespace video {

struct PriorityStreamModule {
  std::chrono::steady_clock::time_point enqueueTime;
  std::condition_variable *cv;

  PriorityStreamModule(std::chrono::steady_clock::time_point enqueueTime,
                       std::condition_variable *cv)
      : enqueueTime(enqueueTime), cv(cv) {}

  bool operator<(const PriorityStreamModule &other) const {
    // 最长等待时间优先
    return enqueueTime > other.enqueueTime;
  }
};

class VideoDecoderPool {
private:
  static std::unique_ptr<VideoDecoderPool> instance;
  static std::mutex instanceMtx;

  std::list<std::unique_ptr<VideoDecode>> available;
  std::mutex mtx;

  size_t initialSize = DECODER_POOL_SIZE;

  VideoDecoderPool() {
    for (size_t i = 0; i < initialSize; ++i) {
      auto decoder = std::make_unique<VideoDecode>();
      if (decoder->init()) { // 初始化解码器
        available.emplace_back(std::move(decoder));
      } else {
        FLOWENGINE_LOGGER_ERROR("decoder init is failed!");
      }
    }
    FLOWENGINE_LOGGER_INFO(
        "Expected to start {} deocders, {} decoders have been started.",
        initialSize, available.size());
  }

public:
  static VideoDecoderPool &getInstance() {
    if (instance == nullptr) {
      std::lock_guard<std::mutex> lock(instanceMtx);
      // double-checked locking
      if (instance == nullptr) {
        instance.reset(new VideoDecoderPool());
      }
    }
    return *instance;
  }

  std::unique_ptr<VideoDecode> acquire() {

    std::unique_lock<std::mutex> lock(mtx);
    if (available.empty()) {
      return nullptr;
    }

    auto decoder = std::move(available.front());
    available.pop_front();
    return decoder;
  }

  void release(std::unique_ptr<VideoDecode> obj) {
    std::lock_guard<std::mutex> lock(mtx);

    obj->stop();
    available.push_back(std::move(obj));
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
};

class PriorityVideoDecoderPool {
private:
  static std::unique_ptr<PriorityVideoDecoderPool> instance;
  static std::mutex instanceMtx;

  std::list<std::unique_ptr<VideoDecode>> available;
  std::priority_queue<PriorityStreamModule> waitingPriorityQueue;
  std::mutex mtx;

  size_t initialSize = DECODER_POOL_SIZE;

  PriorityVideoDecoderPool() {
    for (size_t i = 0; i < initialSize; ++i) {
      auto decoder = std::make_unique<VideoDecode>();
      if (decoder->init()) { // 初始化解码器
        available.emplace_back(std::move(decoder));
      } else {
        FLOWENGINE_LOGGER_ERROR("decoder init is failed!");
      }
    }
    FLOWENGINE_LOGGER_INFO(
        "Expected to start {} deocders, {} decoders have been started.",
        initialSize, available.size());
  }

public:
  static PriorityVideoDecoderPool &getInstance() {
    if (instance == nullptr) {
      std::lock_guard<std::mutex> lock(instanceMtx);
      // double-checked locking
      if (instance == nullptr) {
        instance.reset(new PriorityVideoDecoderPool());
      }
    }
    return *instance;
  }

  std::unique_ptr<VideoDecode>
  acquire(std::chrono::steady_clock::time_point enqueueTime) {

    std::unique_lock<std::mutex> lock(mtx);
    std::condition_variable cv;

    waitingPriorityQueue.push(PriorityStreamModule{enqueueTime, &cv});
    cv.wait(lock, [this, &cv] {
      return !available.empty() && waitingPriorityQueue.top().cv == &cv;
    });
    waitingPriorityQueue.pop();

    auto decoder = std::move(available.front());
    available.pop_front();

    // 为了避免有资源的情况下，因同时竞争而导致的资源空闲
    if (!waitingPriorityQueue.empty()) {
      waitingPriorityQueue.top().cv->notify_one();
    }

    return decoder;
  }

  void release(std::unique_ptr<VideoDecode> obj) {
    std::lock_guard<std::mutex> lock(mtx);

    obj->stop();
    available.push_back(std::move(obj));

    if (!waitingPriorityQueue.empty()) {
      // 这里相当于只条件变量通知所在的线程，因为是栈内局部变量
      waitingPriorityQueue.top().cv->notify_one();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
};

class FIFOVideoDecoderPool {
private:
  static std::unique_ptr<FIFOVideoDecoderPool> instance;
  static std::mutex instanceMtx;

  std::list<std::unique_ptr<VideoDecode>> available;
  std::queue<std::condition_variable *> waitingQueue;
  std::mutex mtx;

  size_t initialSize = DECODER_POOL_SIZE;

  FIFOVideoDecoderPool() {
    for (size_t i = 0; i < initialSize; ++i) {
      auto decoder = std::make_unique<VideoDecode>();
      if (decoder->init()) { // 初始化解码器
        available.emplace_back(std::move(decoder));
      } else {
        FLOWENGINE_LOGGER_ERROR("decoder init is failed!");
      }
    }
    FLOWENGINE_LOGGER_INFO(
        "Expected to start {} deocders, {} decoders have been started.",
        initialSize, available.size());
  }

public:
  static FIFOVideoDecoderPool &getInstance() {
    if (instance == nullptr) {
      std::lock_guard<std::mutex> lock(instanceMtx);
      // double-checked locking
      if (instance == nullptr) {
        instance.reset(new FIFOVideoDecoderPool());
      }
    }
    return *instance;
  }

  std::unique_ptr<VideoDecode> acquire() {

    std::unique_lock<std::mutex> lock(mtx);
    std::condition_variable cv;

    waitingQueue.push(&cv);
    cv.wait(lock, [this, &cv] {
      return !available.empty() && waitingQueue.front() == &cv;
    });
    waitingQueue.pop();

    auto decoder = std::move(available.front());
    available.pop_front();

    // 为了避免有资源的情况下，因同时竞争而导致的资源空闲
    if (!waitingQueue.empty()) {
      waitingQueue.front()->notify_one();
    }

    return decoder;
  }

  void release(std::unique_ptr<VideoDecode> obj) {
    std::lock_guard<std::mutex> lock(mtx);

    obj->stop();
    available.push_back(std::move(obj));

    if (!waitingQueue.empty()) {
      // 这里相当于只条件变量通知所在的线程，因为是栈内局部变量
      waitingQueue.front()->notify_one();
    }
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
};

} // namespace video
#endif