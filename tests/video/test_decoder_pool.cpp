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
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <opencv2/imgcodecs.hpp>
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
    std::this_thread::sleep_for(std::chrono::microseconds(500));
  }
};

void run_stream(std::unique_ptr<VideoDecode> &decoder, std::string url,
                int idx);
int main(int argc, char **argv) {
  std::vector<std::string> urls = {
      "rtsp://admin:zkfd123.com@192.168.31.31:554/Streaming/Channels/101",
      "rtsp://admin:zkfd123.com@192.168.31.41:554/Streaming/Channels/101"

  };
  VideoDecoderPool videoPool(1);

  std::vector<std::unique_ptr<VideoDecode>> decoders;
  auto d1 = videoPool.acquire();
  auto t1 = std::thread(run_stream, std::ref(d1), urls.at(0), 0);
  t1.join();
  videoPool.release(std::move(d1));

  auto d2 = videoPool.acquire();
  auto t2 = std::thread(run_stream, std::ref(d2), urls.at(1), 1);
  t2.join();
  videoPool.release(std::move(d2));

  return 0;
}

void run_stream(std::unique_ptr<VideoDecode> &decoder, std::string url,
                int idx) {
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
}
