#include "gflags/gflags.h"
#include "x3/videoSource.hpp"
#include <chrono>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>
DEFINE_string(uri, "", "Specify the url of video.");

using namespace module::utils;
using namespace std::chrono_literals;
int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"
  // "csi://0"
  videoOptions opt{FLAGS_uri, 2560, 1440, 25, 0};
  std::unique_ptr<videoSource> camera = videoSource::Create(opt);
  camera->Open();

  // std::this_thread::sleep_for(20s);
  void *image;
  int count = 500;
  while (count--) {
    camera->Capture(&image);
    if (count % 10 != 0) {
      continue;
    }
    std::cout << count << ": " << camera->GetHeight() << ", "
              << camera->GetWidth() << std::endl;
    cv::Mat nv12_image = cv::Mat(camera->GetHeight() * 3 / 2,
                                 camera->GetWidth(), CV_8UC1, image);
    if (!nv12_image.empty()) {
      cv::imwrite("test_xvideo.jpg", nv12_image);
    }
  }
  return 0;
}