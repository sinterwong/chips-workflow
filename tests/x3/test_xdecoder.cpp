#include "x3/xDecoder.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

using namespace module::utils;

int main(int argc, char **argv) {
  videoOptions opt{
      std::string(
          "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"),
      1920, 1080, 25, 0};
  std::unique_ptr<XDecoder> camera = XDecoder::create(opt);

  void *image;
  int count = 500;
  while (count--) {
    camera->capture(&image);
    std::cout << opt.height << ", " << opt.width << std::endl;
    cv::Mat nv12_image = cv::Mat(opt.height * 3 / 2, opt.width, CV_8UC1, image);
    cv::imwrite("test_xdecoder.jpg", nv12_image);
  }

  return 0;
}