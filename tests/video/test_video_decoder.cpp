#include "gflags/gflags.h"
#include "logger/logger.hpp"
#include <chrono>
#include <opencv2/imgcodecs.hpp>

#include "video/videoManager.hpp"

using namespace video;
using namespace std::chrono_literals;

DEFINE_string(uri, "", "Specify the url of video.");

int main(int argc, char **argv) {
  // assert(initLogger);
  FLOWENGINE_LOGGER_INFO("starting!");

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // "rtsp://admin:zkfd123.com@192.168.31.31:554/Streaming/Channels/101"
  VideoManager vm{FLAGS_uri};

  vm.init();
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");
  vm.run();
  FLOWENGINE_LOGGER_INFO("Video manager is running!");

  int count = 500;
  while (count--) {
    std::cout << count << ": " << vm.getHeight() << ", " << vm.getWidth()
              << std::endl;
    auto nv12_image = vm.getcvImage();
    if (count % 10 != 0) {
      continue;
    }
    if (!nv12_image->empty()) {
      cv::imwrite("test_xvideo.jpg", *nv12_image);
    }
  }

  gflags::ShutDownCommandLineFlags();
  return 0;
}
