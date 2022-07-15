/**
 * @file detect_module.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <memory>
#include <thread>

#include "boostMessage.h"
#include "common/common.hpp"
#include "jetson/jetsonSourceModule.h"
#include "frameDifferenceModule.h"
#include "logger/logger.hpp"
#include "sendOutputModule.h"
#include <gflags/gflags.h>

DEFINE_string(uri, "", "Specify the uri to run the camera.");
DEFINE_string(result_url, "", "Specify the url to send the results.");
DEFINE_string(codec, "h264", "Specify the video decoding mode.");
DEFINE_int32(height, -1, "Specify video height.");
DEFINE_int32(width, -1, "Specify video width.");

using namespace module;

// This function prints the help information for running this sample
void printArgsInfo() {
  std::cout << "--uri: " << FLAGS_uri << std::endl;
  std::cout << "--result_url: " << FLAGS_result_url << std::endl;
  std::cout << "--codec: " << FLAGS_codec << std::endl;
  std::cout << "--height: " << FLAGS_height << std::endl;
  std::cout << "--width: " << FLAGS_width << std::endl;
}

int main(int argc, char **argv) {
  FlowEngineLoggerInit(true, true, true, true);

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  // /*
  common::AlarmInfo resultInfo;

  std::string resultPoseUrl = FLAGS_result_url;

  BoostMessage bus;
  Backend backend(&bus);
  // rtsp://user:passward@114.242.23.39:9201/test
  // /home/wangxt/workspace/projects/flowengine/tests/data/sample_1080p_h264.mp4
  // rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101
  // rtsp://admin:zkfd123.com@114.242.23.39:9302/cam/realmonitor?channel=1&subtype=0
  std::shared_ptr<JetsonSourceModule> cap(new JetsonSourceModule(
      &backend, FLAGS_uri, FLAGS_width, FLAGS_height, FLAGS_codec, "Camera",
      "FrameMessage", {}, {"FrameDifference"}));

  std::shared_ptr<FrameDifferenceModule> detect(
      new FrameDifferenceModule(&backend, "FrameDifference", "FrameMessage", {"Camera"}, {"SendOutput"}));

  std::shared_ptr<SendOutputModule> output(
      new SendOutputModule(&backend, resultPoseUrl, resultInfo, "SendOutput",
                           "FrameMessage", {"FrameDifference"}));

  std::thread th1(&JetsonSourceModule::go, cap);
  std::thread th2(&FrameDifferenceModule::go, detect);
  std::thread th3(&SendOutputModule::go, output);
  th1.join();
  th2.join();
  th3.join();
  // */

  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}

/* 
./test_bg_module \
 --uri=rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101 \
 --result_url=http://114.242.23.39:9400/v1/internal/receive_alarm \
 --codec=h264 \
 --height=1080 \
 --width=1920 \
*/
