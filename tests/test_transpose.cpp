#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

bool resizeInput(cv::Mat &image, bool isScale, int dst_h, int dst_w) {
  if (isScale) {
    int height = image.rows;
    int width = image.cols;
    float ratio = std::min(dst_h * 1.0 / width, dst_w * 1.0 / height);
    int dw = width * ratio;
    int dh = height * ratio;
    cv::resize(image, image, cv::Size(dw, dh));
    cv::copyMakeBorder(image, image, 0, std::max(0, dst_h - dh), 0,
                       std::max(0, dst_w - dw), cv::BORDER_CONSTANT,
                       cv::Scalar(128, 128, 128));
  } else {
    cv::resize(image, image, cv::Size(dst_w, dst_h));
  }
  return true;
}

void hwc_to_chw(cv::InputArray &src, cv::OutputArray &dst) {

  const int src_h = src.rows();
  const int src_w = src.cols();
  const int src_c = src.channels();

  cv::Mat hw_c = src.getMat().reshape(1, src_h * src_w);

  const std::array<int, 3> dims = {src_c, src_h, src_w};
  dst.create(3, &dims[0], CV_MAKETYPE(src.depth(), 1));
  cv::Mat dst_1d = dst.getMat().reshape(1, {src_c, src_h, src_w});

  std::cout << "src_c: " << src_c << ", "
            << "src_h: " << src_h << ", "
            << "src_w: " << src_w << std::endl;

  cv::transpose(hw_c, dst_1d);
}

void process(void *input) {
  cv::Mat image{600, 800, CV_8UC3, input};
  resizeInput(image, false, 1000, 1000);
  cv::Mat chw_image;
  hwc_to_chw(image, chw_image);
}

int main(int argc, char **argv) {

  cv::Mat image = cv::imread(
      "/home/wangxt/workspace/projects/flowengine/tests/data/pedestrian.jpg");

  process(image.data);

  return 0;
}