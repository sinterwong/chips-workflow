#include "backend.h"
#include "boostMessage.h"
#include "module/x3/streamGenerator.h"

#include "gflags/gflags.h"
DEFINE_string(video, "", "Specify the video uri.");

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
      FLAGS_video,  // "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101"
      1920,
      1080,
      0};
  std::shared_ptr<module::StreamGenerator> decoder;
  decoder = std::make_shared<module::StreamGenerator>(&backend, "SunriseDecoder", "stream", cameraConfig);

  std::thread th1(&module::StreamGenerator::go, decoder);
  th1.join();

  FlowEngineLoggerDrop();
  return 0;
}