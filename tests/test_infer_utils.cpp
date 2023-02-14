#include "gflags/gflags.h"
#include "infer/preprocess.hpp"
#include "infer_utils.hpp"
#include "logger/logger.hpp"
#include <chrono>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <thread>
DEFINE_string(image_path, "", "Specify the path of image.");

using namespace std::chrono_literals;
using namespace infer::utils;
int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  cv::Mat image_bgr = cv::imread(FLAGS_image_path);

  FLOWENGINE_LOGGER_INFO("Image size: {}x{}", image_bgr.cols, image_bgr.rows);

  // bgr to rgb
  cv::Mat image_rgb;
  auto test_bgr_to_rgb = [&image_bgr, &image_rgb](void) {
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
  };
  auto bgr_to_rgb_cost = measureTime(test_bgr_to_rgb);
  cv::imwrite("rgb_image.jpg", image_rgb);

  // rgb to nv12
  cv::Mat image_nv12;
  auto test_rgb_to_yv12 = [&image_rgb, &image_nv12](void) {
    RGB2NV12(image_rgb, image_nv12);
  };
  auto rgb_to_nv12_cost = measureTime(test_rgb_to_yv12);
  cv::imwrite("nv12_image.jpg", image_nv12);

  // nv12 croped
  cv::Mat image_nv12_croped;
  auto test_nv12_crop = [&image_nv12, &image_nv12_croped]() {
    cv::Rect2i roi = {500, 500, 500, 500};
    cropImage(image_nv12, image_nv12_croped, roi, common::ColorType::NV12);
  };
  auto nv12_crop_cost = measureTime(test_nv12_crop);

  cv::imwrite("nv12_croped_image.jpg", image_nv12_croped);
  cv::Mat image_croped_bgr;
  cv::cvtColor(image_nv12_croped, image_croped_bgr, CV_YUV2BGR_NV12);
  cv::imwrite("brg_croped_image.jpg", image_croped_bgr);

  FLOWENGINE_LOGGER_INFO("BRG to RGB cost time: {} ms",
                         static_cast<float>(bgr_to_rgb_cost) / 1000);
  FLOWENGINE_LOGGER_INFO("RGB to NV12 cost time: {} ms",
                         static_cast<float>(rgb_to_nv12_cost) / 1000);
  FLOWENGINE_LOGGER_INFO("NV12 crop image cost time: {} ms",
                         static_cast<float>(nv12_crop_cost) / 1000);
  return 0;
}