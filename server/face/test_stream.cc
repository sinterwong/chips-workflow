/**
 * @file test_stream.cc
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-31
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "algoManager.hpp"
#include "logger/logger.hpp"
#include "streamManager.hpp"
#include <thread>

using namespace server::face;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

int main(int argc, char **argv) {
  FlowEngineLoggerSetLevel(1);

  bool ret;
  ret = core::StreamManager::getInstance().registered(
      "video1", "temp",
      "rtsp://admin:zkfd123.com@localhost:9303/Streaming/Channels/101", "");
  if (!ret) {
    return -1;
  }

  std::this_thread::sleep_for(5s);

  ret = core::StreamManager::getInstance().unregistered("video1");
  if (!ret) {
    return -2;
  }
  return 0;
}