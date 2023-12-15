/**
 * @file test_decoder_pool.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-08-25
 *
 * @copyright Copyright (c) 2023
 *
 */

// 对象池用来管理VideoDecode类
#include "logger/logger.hpp"
#include "videoDecode.hpp"
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

const bool initLogger = []() {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

using video::VideoDecode;

class VideoDecoderPool {
private:
  std::list<std::unique_ptr<VideoDecode>> available;
  std::mutex mtx;

public:
  VideoDecoderPool(size_t initialSize = 0) {
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

  std::unique_ptr<VideoDecode> acquire() {
    std::lock_guard<std::mutex> lock(mtx);
    if (available.empty()) {
      return nullptr;
    }
    auto decoder = std::move(available.back());
    available.pop_back();
    return decoder;
  }

  void release(std::unique_ptr<VideoDecode> obj) {
    std::lock_guard<std::mutex> lock(mtx);

    obj->stop();
    available.push_back(std::move(obj));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
};

class FIFOVideoDecoderPool {
private:
  std::list<std::unique_ptr<VideoDecode>> available;
  std::queue<std::condition_variable *> waitingQueue;
  std::mutex mtx;

public:
  FIFOVideoDecoderPool(size_t initialSize) {
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
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
};

template <typename Pool> void run_stream(Pool &pool, std::string url, int idx);

/**
 * @brief 普通解码对象池测试
 *
 */
void test_decoder_pool(std::vector<std::string> &urls) {
  VideoDecoderPool videoPool(1);

  auto t1 = std::thread(run_stream<VideoDecoderPool>, std::ref(videoPool),
                        urls.at(0), 0);
  t1.join();

  auto d2 = videoPool.acquire();
  auto t2 = std::thread(run_stream<VideoDecoderPool>, std::ref(videoPool),
                        urls.at(1), 1);
  t2.join();
}

/**
 * @brief 先进先出解码池对象测试
 *
 */
void test_fifo_decoder_pool(std::vector<std::string> &urls) {
  FIFOVideoDecoderPool videoPool(2);
  std::vector<std::thread> threads;

  for (size_t i = 0; i < urls.size(); ++i) {
    threads.emplace_back(run_stream<FIFOVideoDecoderPool>, std::ref(videoPool),
                         urls.at(i), i);
  }

  for (auto &t : threads) {
    t.join();
  }
}

int main(int argc, char **argv) {
  std::vector<std::string> urls = {
      "rtsp://admin:zkfd123.com@localhost:554/Streaming/Channels/101",
      "rtsp://admin:zkfd123.com@localhost:554/Streaming/Channels/101",
      "rtsp://admin:zkfd123.com@localhost:554/Streaming/Channels/101"};

  // test_decoder_pool(urls);

  test_fifo_decoder_pool(urls);
  return 0;
}

template <typename Pool>
void run_stream(Pool &pool, std::string url, int idx) {
  auto decoder = pool.acquire();
  if (!decoder->start(url)) {
    FLOWENGINE_LOGGER_INFO("{} start failed!", idx);
    return;
  }
  FLOWENGINE_LOGGER_INFO("{} is running!", idx);

  std::string savePath = std::to_string(idx) + "_test_decoder_pool.jpg";
  int count = 0;
  while (count < 40) {
    if (count % 20 == 0) {
      std::cout << count << ": " << decoder->getHeight() << ", "
                << decoder->getWidth() << std::endl;
      FLOWENGINE_LOGGER_INFO("id: {}, height: {}, width: {}, count: {}", idx,
                             decoder->getHeight(), decoder->getWidth(), count);
    }
    auto nv12_image = decoder->getcvImage();
    if (nv12_image && !nv12_image->empty()) {
      cv::imwrite(savePath, *nv12_image);
    } else {
      FLOWENGINE_LOGGER_ERROR("{} get image was failed!", idx);
    }
    ++count;
  }
  pool.release(std::move(decoder));
}
