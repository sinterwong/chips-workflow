#include "gflags/gflags.h"
#include <cstddef>
#include <cstdlib>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <sp_vio.h>

#include "utils/time_utils.hpp"
#include "logger/logger.hpp"
#include "preprocess.hpp"

using utils::measureTime;
using infer::utils::RGB2NV12;

DEFINE_string(uri, "", "Specify the uri image.");

using namespace std::chrono_literals;
int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"
  // "csi://0"
  cv::Mat image_bgr = cv::imread(FLAGS_uri);
  cv::Mat image_rgb;
  cv::cvtColor(image_bgr, image_rgb, CV_BGR2RGB);

  cv::Mat image_nv12;
  RGB2NV12(image_rgb, image_nv12);

  int dst_h = 512;
  int dst_w = 512;
  int crop_x = 0;
  int crop_y = 0;
  int crop_w = 600;
  int crop_h = 600;
  char *resized_data = reinterpret_cast<char *>(malloc(dst_h * 3 / 2 * dst_w));

  // init vps
  void *vps = sp_init_vio_module();
  // 此处首先执行crop，然后进行resize操作。缺点是图片的输入尺寸初始化的时候就需要确定下来。
  auto test_vps = [&]() {
    // resize只能缩放，如果遇到比resize小的尺寸会报错
    sp_open_vps(vps, 0, 1, SP_VPS_SCALE_CROP, image_bgr.cols, image_bgr.rows,
                &dst_w, &dst_h, &crop_x, &crop_y, &crop_w, &crop_h, NULL);

    int ret = sp_vio_set_frame(vps, image_nv12.data,
                               image_nv12.rows * image_nv12.cols);
    if (ret != 0) {
      FLOWENGINE_LOGGER_ERROR("[Error] sp_vio_set_frame from vps failed!");
      exit(-1);
    }
    // Get the processed image
    ret = sp_vio_get_frame(vps, resized_data, dst_w, dst_h, 2000);
    if (ret != 0) {
      FLOWENGINE_LOGGER_ERROR("[Error] sp_vio_get_frame from vps failed!");
      exit(-1);
    }
    sp_vio_close(vps);
  };
  for (int i = 0; i < 1000; i++) {
    auto vps_cost = measureTime(test_vps);
    FLOWENGINE_LOGGER_INFO("VPS resize and crop cost time: {} ms",
                           static_cast<float>(vps_cost) / 1000);
  }

  cv::Mat resized_image = cv::Mat(dst_h * 3 / 2, dst_w, CV_8UC1, resized_data);
  cv::imwrite("test_vps_resized_image.jpg", resized_image);

  sp_release_vio_module(vps);

  return 0;
}