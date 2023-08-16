#include <gflags/gflags.h>

#include "backend.h"
#include "boostMessage.h"
#include "frameDifferenceModule.h"
#include "infer/algorithmManager.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include "streamModule.h"

DEFINE_string(uri, "", "Specify the url of video.");
DEFINE_double(thre, 0.01, "Specify the threshold for frame diff.");

const bool initLogger = []() {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  module::backend_ptr backendPtr = std::make_shared<Backend>(
      std::make_unique<BoostMessage>(), std::make_unique<StreamPoolBus>(),
      std::make_unique<infer::AlgorithmManager>());

  module::ModuleConfig streamConfig; // 用于初始化 ModuleConfig::configMap
  common::StreamBase streamBase{0,      1920,   1080,      FLAGS_uri,
                                "h264", "rtsp", "myStream"};
  streamConfig.setParams(std::move(streamBase));
  module::StreamModule streamModule{backendPtr, "stream", MessageType::Stream,
                                    streamConfig};

  module::ModuleConfig frameDiffConfig;
  common::AttentionArea aarea;
  common::LogicBase logic_;
  common::AlarmBase alarm_;
  float threshold_ = FLAGS_thre;
  common::DetClsMonitor dcMonitor{std::move(aarea), std::move(logic_),
                                  std::move(alarm_), threshold_};
  frameDiffConfig.setParams(std::move(dcMonitor));
  module::FrameDifferenceModule fdModule{backendPtr, "frameDiff",
                                         MessageType::None, frameDiffConfig};
  streamModule.addSendModule("frameDiff");
  fdModule.addRecvModule("stream");

  joining_thread steamThread([&streamModule]() {
    FLOWENGINE_LOGGER_INFO("streamModule go!");
    streamModule.go();
  });

  joining_thread statusThread([&fdModule]() {
    FLOWENGINE_LOGGER_INFO("fdModule go!");
    fdModule.go();
  });

  gflags::ShutDownCommandLineFlags();
  // FlowEngineLoggerDrop();
  return 0;
}