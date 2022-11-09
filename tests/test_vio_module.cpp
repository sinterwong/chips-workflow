#include "backend.h"
#include "boostMessage.h"
#include "module.hpp"
#include "module/detectModule.h"
#include "moduleFactory.hpp"
#if TARGET_PLATFORM == x3
#include "module/x3/streamGenerator.h"
#elif TARGET_PLATFORM == jetson
#include "module/jetson/streamGenerator.h"
#endif
#include "gflags/gflags.h"
#include <memory>
DEFINE_string(video, "", "Specify the video uri.");
DEFINE_string(model_path, "", "Specify the model path.");

using namespace module;

int main(int argc, char **argv) {
  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Backend backend{std::make_unique<BoostMessage>(),
                  std::make_unique<RouteFramePool>(8)};
  common::CameraConfig cameraConfig{
      "SunriseDecoder",
      "h264",
      "stream",
      FLAGS_video, // "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"
      1920,
      1080,
      0};

  // std::shared_ptr<Module> decoder = ModuleFactory::createModule<Module>(
  //     "StreamGenerator", &backend, "stream_01", "stream", cameraConfig,
  //     std::vector<std::string>(), std::vector<std::string>{"handDet"});
  std::shared_ptr<Module> decoder = std::make_shared<StreamGenerator>(
      &backend, "stream_01", "stream", cameraConfig, std::vector<std::string>{},
      std::vector<std::string>{"handDet"});

  std::array<int, 3> inputShape{640, 640, 3};
  std::vector<std::string> inputNames = {"images"};
  std::vector<std::string> outputNames = {"output"};
  common::AlgorithmConfig params{FLAGS_model_path,
                                 std::move(inputNames),
                                 std::move(outputNames),
                                 std::move(inputShape),
                                 "yolo",
                                 0.4,
                                 0.4,
                                 255.0,
                                 0,
                                 false,
                                 1};

  // std::shared_ptr<Module> handDet = ModuleFactory::createModule<Module>(
  //     "DetectModule", &backend, "handDet", "algorithm", params,
  //     std::vector<std::string>{"stream_01"}, std::vector<std::string>{});
  std::shared_ptr<Module> handDet = std::make_shared<DetectModule>(
      &backend, "handDet", "algorithm", params,
      std::vector<std::string>{"stream_01"}, std::vector<std::string>{});

  if (handDet == nullptr) {
    std::cout << "handDet"
              << " is nullptr!!!!!!!!!!!!!!!!!!" << std::endl;
    return -1;
  }

  if (decoder == nullptr) {
    std::cout << "stream_01"
              << " is nullptr!!!!!!!!!!!!!!!!!!" << std::endl;
    return -1;
  }

  std::thread th1(&Module::go, decoder);
  std::thread th2(&Module::go, handDet);
  th1.join();
  th2.join();

  FlowEngineLoggerDrop();
  return 0;
}