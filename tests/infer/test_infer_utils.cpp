#include "gflags/gflags.h"
#include "infer/preprocess.hpp"
#include "logger/logger.hpp"
#include "utils/time_utils.hpp"
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
using namespace utils;
using common::Points2i;
using Point2i = common::Point<int>;
using common::RetBox;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

void test_color_convert() {
  cv::Mat image_bgr = cv::imread(FLAGS_image_path);

  FLOWENGINE_LOGGER_INFO("Image size: {}x{}", image_bgr.cols, image_bgr.rows);

  // bgr to rgb
  cv::Mat image_rgb;
  auto test_bgr_to_rgb = [&image_bgr, &image_rgb](void) {
    cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
  };
  auto bgr_to_rgb_cost = measureTime(test_bgr_to_rgb);
  FLOWENGINE_LOGGER_INFO("BRG to RGB cost time: {} ms",
                         static_cast<float>(bgr_to_rgb_cost) / 1000);

  // rgb to nv12
  cv::Mat image_nv12;
  auto test_rgb_to_yv12 = [&image_rgb, &image_nv12](void) {
    RGB2NV12(image_rgb, image_nv12);
  };
  auto rgb_to_nv12_cost = measureTime(test_rgb_to_yv12);
  FLOWENGINE_LOGGER_INFO("RGB to NV12 cost time: {} ms",
                         static_cast<float>(rgb_to_nv12_cost) / 1000);

  // nv12 croped
  cv::Mat image_nv12_croped;
  auto test_nv12_crop = [&image_nv12, &image_nv12_croped]() {
    cv::Rect2i roi = {0, 0, 1920, 1080};
    cropImage(image_nv12, image_nv12_croped, roi, common::ColorType::NV12);
  };
  auto nv12_crop_cost = measureTime(test_nv12_crop);
  FLOWENGINE_LOGGER_INFO("NV12 crop image cost time: {} ms",
                         static_cast<float>(nv12_crop_cost) / 1000);

  cv::Mat image_nv12_resized;
  for (int i = 0; i < 10; i++) {
    // // nv12 resize
    // cv::Mat image_nv12_resized;
    // auto test_nv12_resize = [&image_nv12_croped, &image_nv12_resized]() {
    //   cv::Mat temp;
    //   NV12toRGB(image_nv12_croped, temp);
    //   std::array<int, 2> shape{640, 640};
    //   resizeInput(temp, false, shape);
    //   RGB2NV12(temp, image_nv12_resized);
    // };
    // auto nv12_resize_cost = measureTime(test_nv12_resize);
    // FLOWENGINE_LOGGER_INFO("NV12 resize image cost time: {} ms",
    //                        static_cast<float>(nv12_resize_cost) / 1000);

    // nv12 resize
    auto test_nv12_resize = [&image_nv12_croped, &image_nv12_resized]() {
      std::array<int, 2> shape{640, 640};
      int y_rows = image_nv12_croped.rows * 2 / 3;
      cv::Mat y = image_nv12_croped.rowRange(0, y_rows).colRange(
          0, image_nv12_croped.cols);
      cv::Mat uv = image_nv12_croped.rowRange(y_rows, image_nv12_croped.rows)
                       .colRange(0, image_nv12_croped.cols);

      // 将UV分别重塑为两个矩阵
      cv::Mat u = uv.rowRange(0, uv.rows / 2).colRange(0, uv.cols / 2);
      cv::Mat v = uv.rowRange(uv.rows / 2, uv.rows).colRange(0, uv.cols / 2);

      cv::Mat y_resized, u_resized, v_resized;
      cv::resize(y, y_resized, cv::Size(shape.at(0), shape.at(1)));
      cv::resize(u, u_resized, cv::Size(shape.at(0) / 2, shape.at(1)) / 2);
      cv::resize(v, v_resized, cv::Size(shape.at(0) / 2, shape.at(1)) / 2);
      // cv::imwrite("y_resized.jpg", y_resized);
      // cv::imwrite("u_resized.jpg", u_resized);
      // cv::imwrite("v_resized.jpg", v_resized);

      image_nv12_resized.create(cv::Size(shape.at(0), shape.at(1) * 3 / 2),
                                CV_8UC1);
      y_resized.copyTo(image_nv12_resized.rowRange(0, y_resized.rows)
                           .colRange(0, y_resized.cols));
      // 将重采样后的U和V分量并入一个矩阵中
      cv::Mat uvInterleaved =
          image_nv12_resized.rowRange(y_resized.rows, image_nv12_resized.rows)
              .colRange(0, image_nv12_resized.cols);

      // 将U和V矩阵分别复制到UV通道中，每个U/V对应一个2x2的位置
      for (int row = 0; row < uvInterleaved.rows / 2; ++row) {
        uchar *uvInterleavedRow = uvInterleaved.ptr<uchar>(2 * row);
        uchar *uRow = u_resized.ptr<uchar>(row);
        uchar *vRow = v_resized.ptr<uchar>(row);

        for (int col = 0; col < uvInterleaved.cols / 2; ++col) {
          uvInterleavedRow[2 * col] = uRow[col];
          uvInterleavedRow[2 * col + 1] = vRow[col];
        }

        uvInterleavedRow = uvInterleaved.ptr<uchar>(2 * row + 1);

        for (int col = 0; col < uvInterleaved.cols / 2; ++col) {
          uvInterleavedRow[2 * col] = uRow[col];
          uvInterleavedRow[2 * col + 1] = vRow[col];
        }
      }
    };
    auto nv12_resize_cost = measureTime(test_nv12_resize);
    FLOWENGINE_LOGGER_INFO("NV12 resize image cost time: {} ms",
                           static_cast<float>(nv12_resize_cost) / 1000);
  }

  cv::imwrite("rgb_image.jpg", image_rgb);
  cv::imwrite("nv12_image.jpg", image_nv12);
  cv::imwrite("nv12_croped_image.jpg", image_nv12_croped);
  cv::imwrite("nv12_resized_image.jpg", image_nv12_resized);

  cv::Mat image_resized_bgr;
  cv::cvtColor(image_nv12_resized, image_resized_bgr, CV_YUV2BGR_NV12);
  cv::imwrite("bgr_resized_image.jpg", image_resized_bgr);
}

void test_new_crop() {
  cv::Mat image_bgr = cv::imread(FLAGS_image_path);
  FLOWENGINE_LOGGER_INFO("Image size: {}x{}", image_bgr.cols, image_bgr.rows);
  // bgr to rgb
  cv::Mat image_rgb, image_rgb_croped;
  cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);
  // rgb to nv12
  cv::Mat image_nv12, image_nv12_croped, image_nv12_croped_bgr;
  RGB2NV12(image_rgb, image_nv12);

  common::Points2i points{Point2i{100, 100}, Point2i{300, 100},
                          Point2i{350, 400}, Point2i{80, 450}};

  RetBox bbox{"hello", points};

  cropImage(image_rgb, image_rgb_croped, bbox, common::ColorType::RGB888, 0.5);
  cv::imwrite("test_new_crop_rgb.jpg", image_rgb_croped);

  cropImage(image_nv12, image_nv12_croped, bbox, common::ColorType::NV12, 0.5);
  cv::imwrite("test_new_crop_nv12.jpg", image_nv12_croped);

  cv::cvtColor(image_nv12_croped, image_nv12_croped_bgr, cv::COLOR_YUV2BGR_NV12);
  cv::imwrite("test_new_crop_nv12_bgr.jpg", image_nv12_croped_bgr);
}

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // test_color_convert();

  test_new_crop();

  return 0;
}