#include "logger/logger.hpp"
#include <gflags/gflags.h>

DEFINE_string(uri, "", "Specify the uri of video.");

int main(int argc, char **argv) {
  // FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);


  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}
