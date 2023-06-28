#include "ffstream.hpp"
#include "logger/logger.hpp"
#include "time_utils.hpp"
#include <chrono>
#include <gflags/gflags.h>
#include <iostream>
DEFINE_string(uri, "", "Specify the url of video.");

const bool initLogger = []() {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 测试ffstream内存泄露情况
  video::utils::FFStream stream{FLAGS_uri};
  if (!stream.openStream()) {
    FLOWENGINE_LOGGER_ERROR("open stream failed!");
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("open stream success!");
  void *data = nullptr;

  auto time = utils::measureTime([&]() {
  int count = 0;
    while (stream.isRunning() && ++count < 1000) {
      if (count % 100 == 0) {
        FLOWENGINE_LOGGER_INFO("count: {}", count);
      }
      int bufSize = stream.getRawFrame(&data, false);
      if (bufSize < 0) {
        break;
      }
    }
  });
  FLOWENGINE_LOGGER_INFO("time: {} ms", static_cast<double>(time) / 1000);

  if (stream.isRunning()) {
    stream.closeStream();
  }
  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}