#include "gflags/gflags.h"
#include "module/configParser.hpp"
DEFINE_string(config_path, "", "Specify the path of image.");

using namespace module::utils;

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ConfigParser parser;
  std::vector<PipelineParams> pipelines;

  parser.parseConfig(FLAGS_config_path, pipelines);

  for (auto &pipe : pipelines) {
    for (auto &module : pipe) {
      std::cout << module.first << std::endl;
      // TODO 检查参数提取是否有问题，没问题了开始更新pipeline
    }
  }

  gflags::ShutDownCommandLineFlags();

  return 0;
}
