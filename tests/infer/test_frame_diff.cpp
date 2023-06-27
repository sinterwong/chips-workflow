/**
 * @file test_frame_diff.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-06-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "frame_difference.h"
#include "logger/logger.hpp"
#include "preprocess.hpp"
#include "videoDecode.hpp"
#include <future>
#include <gflags/gflags.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <utility>

DEFINE_string(uri, "", "Specify the url of video.");

using namespace infer::solution;

const bool initLogger = []() {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FrameDifference frame_diff;

  // 测试decoder内存泄露情况
  video::VideoDecode decoder{FLAGS_uri};

  if (!decoder.init()) {
    FLOWENGINE_LOGGER_ERROR("init decoder failed!");
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");

  if (!decoder.run()) {
    FLOWENGINE_LOGGER_ERROR("run decoder failed!");
    return -1;
  }

  while (1) {
    auto frame = decoder.getcvImage();
    cv::Mat input;
    infer::utils::NV12toRGB(*frame, input);
    if (frame->empty()) {
      FLOWENGINE_LOGGER_INFO("Video end!");
      break;
    }
    std::vector<RetBox> bboxes;
    frame_diff.update(input, bboxes);
    for (auto &bbox : bboxes) {
      FLOWENGINE_LOGGER_INFO("bbox: {},{},{},{},{}", bbox.first, bbox.second[0],
                             bbox.second[1], bbox.second[2], bbox.second[3]);
      // draw bbox in frame
      cv::rectangle(input,
                    cv::Rect(bbox.second[0], bbox.second[1],
                             bbox.second[2] - bbox.second[0],
                             bbox.second[3] - bbox.second[1]),
                    cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("draw.jpg", input);
  }
  gflags::ShutDownCommandLineFlags();
  return 0;
}
