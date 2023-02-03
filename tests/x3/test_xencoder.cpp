#include "gflags/gflags.h"
#include "logger/logger.hpp"
#include "x3/xEncoder.hpp"
#include "x3/xCamera.hpp"
#include <chrono>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>
DEFINE_string(uri, "output.h264", "Specify the url of video.");

using namespace module::utils;
using namespace std::chrono_literals;
int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  videoOptions opt1{FLAGS_uri, 1920, 1080, 25, 0};
  std::unique_ptr<XCamera> camera = XCamera::create(opt1);

  // "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"
  // "csi://0"
  videoOptions opt{FLAGS_uri, 1920, 1080, 25, 1};
  std::unique_ptr<XEncoder> output = XEncoder::create(opt);
  output->init();

  void *image;
  int count = 250;
  while (count--) {
    if(!camera->capture(&image)) {
      FLOWENGINE_LOGGER_CRITICAL("capture is failed!");
    }
    if(!output->render(&image)) {
      FLOWENGINE_LOGGER_CRITICAL("render is failed!");
    };
  }
  FLOWENGINE_LOGGER_INFO("Done!");
  return 0;
}