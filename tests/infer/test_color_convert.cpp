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

  gflags::ShutDownCommandLineFlags();

  return 0;
}

/*
./test_face --img1 /root/workspace/softwares/flowengine/data/face1.jpg \
            --img2 /root/workspace/softwares/flowengine/data/face2.jpg \
            --model_path
/root/workspace/softwares/flowengine/models/facenet_mobilenet_160x160.bin
*/