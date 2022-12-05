#include "gflags/gflags.h"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include "pipeline.hpp"
#include "x3/videoManager.hpp"
#include <cassert>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <thread>

using namespace module;
using namespace std::chrono_literals;

int main(int argc, char **argv) {
  // assert(initLogger);
  FLOWENGINE_LOGGER_INFO("starting!");

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::shared_ptr<PipelineModule> pipeline =
      std::make_shared<PipelineModule>("hello", 0);
  // "rtsp://admin:zkfd123.com@192.168.31.31:554/Streaming/Channels/101"
  module::utils::VideoManager vm{
      "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101", 0};

  vm.init();
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");
  vm.run();
  FLOWENGINE_LOGGER_INFO("Video manager is running!");
  std::this_thread::sleep_for(30ms);
  auto image = vm.getcvImage();
  FLOWENGINE_LOGGER_CRITICAL("Saving image..");
  cv::imwrite("vm_out.jpg", *image);
  FLOWENGINE_LOGGER_CRITICAL("I'm Here");
  auto logger_ptr = spdlog::get(FLOWENGINE_LOGGER_NAME);
  if (!logger_ptr) {
    FlowEngineLoggerDrop();
  }
  return 0;
}
