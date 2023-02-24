#include "gflags/gflags.h"
#include "x3/videoManager.hpp"
#include "x3/videoSource.hpp"
#include <chrono>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

DEFINE_string(uri, "", "Specify the url of video.");

using namespace module::utils;
using namespace std::chrono_literals;

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"
  // "csi://0"
  VideoManager vm{FLAGS_uri};
  vm.init();
  vm.run();

  std::this_thread::sleep_for(200ms);

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
  return 0;
}