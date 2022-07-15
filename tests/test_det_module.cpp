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
#include "jetson/detectModule.h"
#include "jetson/jetsonSourceModule.h"
#include "logger/logger.hpp"
#include "sendOutputModule.h"
#include <gflags/gflags.h>

DEFINE_int32(height, -1, "Specify video height.");
DEFINE_int32(width, -1, "Specify video width.");
DEFINE_string(uri, "", "Specify the uri to run the camera.");
DEFINE_string(result_url, "", "Specify the url to send the results.");
DEFINE_string(codec, "h264", "Specify the video decoding mode.");
DEFINE_string(model_dir, "", "Specify the dir to models.");

DEFINE_string(configs, "",
              "Configuration of all algorithms that need to be started");

using namespace module;

// This function prints the help information for running this sample
void printArgsInfo() {
  std::cout << "--model_dir: " << FLAGS_model_dir << std::endl;
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
  // TODO 想办法做好初始化
  // infer::InferParams params;
  // params.batchSize = 1;
  // params.numAnchors = 25200;
  // params.numClasses = 80;
  // params.inputTensorNames.push_back("images");
  // params.outputTensorNames.push_back("output");
  // params.inputShape = {640, 640, 3};
  // params.serializedFilePath = FLAGS_model_dir + "/yolov5s.engine";
  // params.originShape = {FLAGS_width, FLAGS_height, 3};
  // params.scaling =
  //     float(params.inputShape[0]) /
  //     (float)std::max(params.originShape[0], params.originShape[1]);

  common::ParamsConfig params;
  params.originShape = {FLAGS_width, FLAGS_height, 3};
  params.cond_thr = 0.3;
  params.nms_thr = 0.3;
  params.modelDir = FLAGS_model_dir;

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
      "FrameMessage", {}, {"Detection"}));

  std::shared_ptr<DetectModule> detect(
      new DetectModule(&backend, "Detection", "FrameMessage", params,
                       {"Camera"}, {"SendOutput"}));

  std::shared_ptr<SendOutputModule> output(
      new SendOutputModule(&backend, resultPoseUrl, resultInfo, "SendOutput",
                           "FrameMessage", {"Detection"}));

  std::thread th1(&JetsonSourceModule::go, cap);
  std::thread th2(&DetectModule::go, detect);
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
./test_det_module \
 --model_dir=/home/wangxt/workspace/projects/flowengine/tests/data \
 --uri=rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101 \
 --result_url=http://114.242.23.39:9400/v1/internal/receive_alarm \
 --codec=h264 \
 --height=1920 \
 --width=1080 \
*/
