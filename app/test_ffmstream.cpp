#include "ffstream.hpp"
#include "logger/logger.hpp"
#include "time_utils.hpp"
#include <chrono>
#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
  if (!stream.openStream(true)) {
    FLOWENGINE_LOGGER_ERROR("open stream failed!");
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("open stream success!");
  void *data = nullptr;

  auto time = utils::measureTime([&]() {
    int count = 0;
    while (stream.isRunning() && ++count < 1000) {

      int bufSize = stream.getDataFrame(&data, false, false);
      if (bufSize < 0) {
        FLOWENGINE_LOGGER_ERROR("UNKNOWN FAILED.");
        break;
      } else if (bufSize == 0) {
        // 当前帧失败
        FLOWENGINE_LOGGER_WARN("current frame failed.");
        continue;
      }
      // use opencv to write data to file
      if (count % 25 == 0) {
        FLOWENGINE_LOGGER_INFO("count: {}", count);
        cv::Mat frame{stream.getHeight(), stream.getWidth(), CV_8UC3, data};
        if (frame.empty()) {
          FLOWENGINE_LOGGER_ERROR("frame is empty!");
          break;
        }
        cv::imwrite("test_ffmstream_frame.jpg", frame);
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