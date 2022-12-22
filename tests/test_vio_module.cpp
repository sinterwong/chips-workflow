#include "backend.h"
#include "boostMessage.h"
#include "common/config.hpp"
#include "factory.hpp"
#include "gflags/gflags.h"
#include "module.hpp"
#include "pipeline.hpp"
#include <memory>
using namespace module;

DEFINE_string(video, "", "Specify the video uri.");
DEFINE_string(model_path, "", "Specify the model path.");

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 因为动态链接的问题，因此这里需要用一下库里面的东西
  // std::shared_ptr<PipelineModule> pipeline =
  //     std::make_shared<PipelineModule>("hello", 0);

  Backend backend{std::make_unique<BoostMessage>(),
                  std::make_unique<RouteFramePool>(8)};
  // "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"
  common::CameraConfig cameraConfig{
      "StreamName", "h264", "stream", FLAGS_video, 1920, 1080, 0};
  std::shared_ptr<Module> decoder = ObjectFactory::createObject<Module>(
      "StreamModule", &backend, std::string("stream_01"),
      std::string("stream"), cameraConfig);
  if (decoder == nullptr) {
    std::cout << "stream_01"
              << " is nullptr!!!!!!!!!!!!!!!!!!" << std::endl;
    return -1;
  }

  // 算法module
  std::array<int, 3> inputShape{640, 640, 3};
  std::vector<std::string> inputNames = {"images"};
  std::vector<std::string> outputNames = {"output"};
  common::AlgorithmConfig params{FLAGS_model_path,
                                 std::move(inputNames),
                                 std::move(outputNames),
                                 std::move(inputShape),
                                 "Yolo",
                                 0.4,
                                 0.4,
                                 255.0,
                                 0,
                                 false,
                                 1};

  std::shared_ptr<Module> handDet = ObjectFactory::createObject<Module>(
      "DetectModule", &backend, std::string("handDet"),
      std::string("algorithm"), params);
  if (handDet == nullptr) {
    std::cout << "handDet"
              << " is nullptr!!!!!!!!!!!!!!!!!!" << std::endl;
    return -1;
  }

  decoder->addSendModule("handDet");
  handDet->addRecvModule("stream_01");

  std::thread th1(&Module::go, decoder);
  std::thread th2(&Module::go, handDet);
  th1.join();
  th2.join();
  return 0;
}