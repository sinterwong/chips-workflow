#include "logger/logger.hpp"
#include "time_utils.hpp"
#include "videoDecode.hpp"
#include <chrono>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
DEFINE_string(uri, "", "Specify the url of video.");

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 测试decoder内存泄露情况
  video::VideoDecode decoder{FLAGS_uri};
  if (!decoder.init()) {
    FLOWENGINE_LOGGER_ERROR("init decoder failed!");
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");

  if (!decoder.run()) {
    FLOWENGINE_LOGGER_ERROR("run decoder failed!");
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("Video manager is running!");

  auto time = utils::measureTime([&]() {
    int count = 0;
    while (decoder.isRunning() && ++count < 1000) {
      auto image = decoder.getcvImage();
      if (count % 30 == 0) {
        FLOWENGINE_LOGGER_INFO("count: {}", count);
        if (!image) {
          FLOWENGINE_LOGGER_ERROR("get image failed!");
          continue;
        }
        cv::imwrite("test_decoder_out.jpg", *image);
      }
    }
  });
  FLOWENGINE_LOGGER_INFO("total time: {} ms", static_cast<double>(time) / 1000);
  gflags::ShutDownCommandLineFlags();
  return 0;
}