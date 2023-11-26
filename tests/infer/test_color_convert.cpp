#include <gflags/gflags.h>
#include <iostream>

#include "logger/logger.hpp"
#include "utils/time_utils.hpp"

#include "infer/preprocess.hpp"

#include <opencv2/opencv.hpp>

#include <arm_neon.h>

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

DEFINE_string(img_path, "", "Specify image path.");
DEFINE_int32(times, 10, "Specify test times.");

void RGB2NV12(const cv::Mat &input, cv::Mat &output) {
  // 保证宽度和高度都是偶数
  int rows = (input.rows >> 1) << 1;
  int cols = (input.cols >> 1) << 1;

  // 分配输出矩阵的空间
  output.create(rows + rows / 2, cols, CV_8UC1);

  // Y通道和UV通道的指针
  uchar *Y = output.data;
  uchar *UV = Y + (rows * cols);

  for (int y = 0; y < rows; y++) {
    const uchar *input_ptr = input.ptr<uchar>(y);

    for (int x = 0; x < cols; x++) {
      int b = *input_ptr++;
      int g = *input_ptr++;
      int r = *input_ptr++;

      // 计算Y通道
      int y_value = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
      Y[y * cols + x] = cv::saturate_cast<uchar>(y_value);

      // 计算UV通道，仅在偶数行和偶数列处计算
      if (y % 2 == 0 && x % 2 == 0) {
        int u_value = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        int v_value = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        UV[(y / 2) * cols + x] = cv::saturate_cast<uchar>(u_value);
        UV[(y / 2) * cols + x + 1] = cv::saturate_cast<uchar>(v_value);
      }
    }
  }
}

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto image_bgr = cv::imread(FLAGS_img_path);
  FLOWENGINE_LOGGER_INFO("Image size: {}x{}", image_bgr.cols, image_bgr.rows);

  cv::Mat image_rgb, image_nv12_a, image_nv12_b, image_nv12_c;
  cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

  long long time_ori = 0;
  long long time_parallel = 0;
  long long time_new = 0;
  for (int i = 0; i < FLAGS_times; ++i) {
    time_parallel += utils::measureTime(
        [&]() { infer::utils::RGB2NV12(image_rgb, image_nv12_a, true); });
  }
  cv::imwrite("test_color_convert_nv12_a.jpg", image_nv12_a);
  FLOWENGINE_LOGGER_INFO("parallel total time: {} ms, avg time: {} ms",
                         static_cast<double>(time_parallel) / 1000,
                         static_cast<double>(time_parallel) / 1000 /
                             FLAGS_times);

  for (int i = 0; i < FLAGS_times; ++i) {
    time_ori += utils::measureTime(
        [&]() { infer::utils::RGB2NV12(image_rgb, image_nv12_b, false); });
  }
  cv::imwrite("test_color_convert_nv12_b.jpg", image_nv12_b);
  FLOWENGINE_LOGGER_INFO("original total time: {} ms, avg time: {} ms",
                         static_cast<double>(time_ori) / 1000,
                         static_cast<double>(time_ori) / 1000 / FLAGS_times);

  for (int i = 0; i < FLAGS_times; ++i) {
    time_new +=
        utils::measureTime([&]() { RGB2NV12(image_rgb, image_nv12_c); });
  }
  cv::imwrite("test_color_convert_nv12_c.jpg", image_nv12_c);
  FLOWENGINE_LOGGER_INFO("new total time: {} ms, avg time: {} ms",
                         static_cast<double>(time_new) / 1000,
                         static_cast<double>(time_new) / 1000 / FLAGS_times);
  gflags::ShutDownCommandLineFlags();

  return 0;
}

/*
./test_face --img1 /root/workspace/softwares/flowengine/data/face1.jpg \
            --img2 /root/workspace/softwares/flowengine/data/face2.jpg \
            --model_path
/root/workspace/softwares/flowengine/models/facenet_mobilenet_160x160.bin
*/