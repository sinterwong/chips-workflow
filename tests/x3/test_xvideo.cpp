#include "gflags/gflags.h"
#include "x3/videoSource.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
DEFINE_string(uri, "", "Specify the url of video.");

using namespace module::utils;
int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"
  // "csi://0"
  videoOptions opt{FLAGS_uri, 1920, 1080, 25, 0};
  std::unique_ptr<videoSource> camera = videoSource::create(opt);

  void *image;
  int count = 500;
  while (count--) {
    camera->capture(&image);
    std::cout << opt.height << ", " << opt.width << std::endl;
    cv::Mat nv12_image = cv::Mat(opt.height * 3 / 2, opt.width, CV_8UC1, image);
    cv::imwrite("test_xvideo.jpg", nv12_image);
  }
  return 0;
}