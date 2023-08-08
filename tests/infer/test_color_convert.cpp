#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "logger/logger.hpp"
#include "utils/time_utils.hpp"

#include "infer/preprocess.hpp"

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

DEFINE_string(img_path, "", "Specify image path.");
DEFINE_int32(times, 10, "Specify test times.");

void YU122NV12(cv::Mat const &input, cv::Mat &output) {
  int y_rows = input.rows * 2 / 3;
  cv::Mat y = input.rowRange(0, y_rows).colRange(0, input.cols);
  cv::Mat uv = input.rowRange(y_rows, input.rows).colRange(0, input.cols);
  output.create(cv::Size(input.cols, input.rows), CV_8UC1);

  y.copyTo(output.rowRange(0, y.rows).colRange(0, y.cols));

  cv::Mat uvInterleaved =
      output.rowRange(y_rows, input.rows).colRange(0, uv.cols);

  cv::Mat u = uv.rowRange(0, uv.rows / 2).colRange(0, uv.cols / 2);
  cv::Mat v = uv.rowRange(uv.rows / 2, uv.rows).colRange(0, uv.cols / 2);

  // Use cv::parallel_for_ to parallelize the filling of uvInterleaved
  cv::parallel_for_(cv::Range(0, uv.rows / 2), [&](const cv::Range &range) {
    for (int row = range.start; row < range.end; ++row) {
      uchar *uvInterleavedRow = uvInterleaved.ptr<uchar>(2 * row);
      uchar *uRow = u.ptr<uchar>(row);
      uchar *vRow = v.ptr<uchar>(row);

      for (int col = 0; col < uv.cols / 2; ++col) {
        uvInterleavedRow[2 * col] = uRow[col];
        uvInterleavedRow[2 * col + 1] = vRow[col];
      }
      uvInterleavedRow = uvInterleaved.ptr<uchar>(2 * row + 1);
      for (int col = 0; col < uv.cols / 2; ++col) {
        uvInterleavedRow[2 * col] = uRow[col];
        uvInterleavedRow[2 * col + 1] = vRow[col];
      }
    }
  });
}

void RGB2NV12_parallel(cv::Mat const &input, cv::Mat &output) {
  cv::Rect rect{0, 0, input.cols, input.rows};
  if (rect.width % 2 != 0)
    rect.width -= 1;
  if (rect.height % 2 != 0) {
    rect.height -= 1;
  }
  cv::Mat in_temp = input(rect);
  cv::Mat temp;
  cv::cvtColor(in_temp, temp, cv::COLOR_RGB2YUV_I420);
  YU122NV12(temp, output);
}

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  auto image_bgr = cv::imread(FLAGS_img_path);
  FLOWENGINE_LOGGER_INFO("Image size: {}x{}", image_bgr.cols, image_bgr.rows);

  cv::Mat image_rgb, image_nv12_a, image_nv12_b;
  cv::cvtColor(image_bgr, image_rgb, cv::COLOR_BGR2RGB);

  long long time_ori = 0;
  long long time_parallel = 0;
  for (int i = 0; i < FLAGS_times; ++i) {
    time_parallel += utils::measureTime(
        [&]() { RGB2NV12_parallel(image_rgb, image_nv12_a); });
  }
  cv::imwrite("test_color_convert_nv12_a.jpg", image_nv12_a);
  FLOWENGINE_LOGGER_INFO("parallel total time: {} ms, avg time: {} ms",
                         static_cast<double>(time_parallel) / 1000,
                         static_cast<double>(time_parallel) / 1000 / FLAGS_times);

  for (int i = 0; i < FLAGS_times; ++i) {
    time_ori += utils::measureTime(
        [&]() { infer::utils::RGB2NV12(image_rgb, image_nv12_b); });
  }
  cv::imwrite("test_color_convert_nv12_b.jpg", image_nv12_b);
  FLOWENGINE_LOGGER_INFO("original total time: {} ms, avg time: {} ms",
                         static_cast<double>(time_ori) / 1000,
                         static_cast<double>(time_ori) / 1000 / FLAGS_times);

  gflags::ShutDownCommandLineFlags();

  return 0;
}

/*
./test_face --img1 /root/workspace/softwares/flowengine/data/face1.jpg \
            --img2 /root/workspace/softwares/flowengine/data/face2.jpg \
            --model_path
/root/workspace/softwares/flowengine/models/facenet_mobilenet_160x160.bin
*/