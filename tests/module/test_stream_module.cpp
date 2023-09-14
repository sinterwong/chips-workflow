#include <gflags/gflags.h>

#include "backend.h"
#include "boostMessage.h"
#include "infer/algorithmManager.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include "statusOutputModule.h"
#include "streamModule.h"

DEFINE_string(uri, "", "Specify the url of video.");

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
  common::StreamBase streamBase{{100},     0,      1920,   1080,
                                FLAGS_uri, "h264", "rtsp", "myStream"};
  streamConfig.setParams(std::move(streamBase));
  module::StreamModule streamModule{backendPtr, "stream", MessageType::Stream,
                                    streamConfig};

  module::ModuleConfig statusConfig;
  common::OutputBase outputBase{{100}, "http://localhost:9876/v1/flow/alarm"};
  statusConfig.setParams(std::move(outputBase));
  module::StatusOutputModule statusOutputModule{
      backendPtr, "statusOutput", MessageType::Status, statusConfig};
  streamModule.addSendModule("statusOutput");
  statusOutputModule.addRecvModule("stream");

  joining_thread steamThread([&streamModule]() {
    FLOWENGINE_LOGGER_INFO("streamModule go!");
    streamModule.go();
  });

  joining_thread statusThread([&statusOutputModule]() {
    FLOWENGINE_LOGGER_INFO("statusOutputModule go!");
    statusOutputModule.go();
  });

  gflags::ShutDownCommandLineFlags();
  // FlowEngineLoggerDrop();
  return 0;
}