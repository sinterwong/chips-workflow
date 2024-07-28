#include "videoSource.hpp"
#include "xCamera.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

int main(int argc, char **argv) {
  video::videoOptions opt{std::string("csi://0"), 1920, 1080, 25, 0};
  std::unique_ptr<video::XCamera> camera = video::XCamera::Create(opt);

  void *image;
  int count = 500;
  while (count--) {
    camera->Capture(&image);
    std::cout << opt.height << ", " << opt.width << std::endl;
    cv::Mat nv12_image = cv::Mat(opt.height * 3 / 2, opt.width, CV_8UC1, image);
    cv::imwrite("test_xcamera.jpg", nv12_image);
  }

  return 0;
}