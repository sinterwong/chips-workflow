/**
 * @file test_priority_decoder_pool.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-09-01
 *
 * @copyright Copyright (c) 2023
 *
 */

// 对象池用来管理VideoDecode类
#include "logger/logger.hpp"
#include "videoDecoderPool.hpp"
#include <opencv2/imgcodecs.hpp>

const bool initLogger = []() {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

template <typename Pool> void run_stream(Pool &pool, std::string url, int idx) {
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

int main(int argc, char **argv) {
  std::vector<std::string> urls = {
      "rtsp://admin:zkfd123.com@localhost:554/Streaming/Channels/101",
      "rtsp://admin:zkfd123.com@localhost:554/Streaming/Channels/101",
      "rtsp://admin:zkfd123.com@localhost:554/Streaming/Channels/101"};

  return 0;
}
