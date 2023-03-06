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
      if (module.first.className == "StreamModule") {
        auto config = module.second.getParams<common::StreamBase>();
        if (!config) {
          std::cout << "error" << std::endl;
          return -1;
        }
        std::cout << "cameraName: " << config->cameraName << std::endl;
        std::cout << "cameraId: " << config->cameraId << std::endl;
        std::cout << "flowType: " << config->flowType << std::endl;
        std::cout << "uri: " << config->uri << std::endl;
        std::cout << "videoCode: " << config->videoCode << std::endl;
        std::cout << "height: " << config->height << std::endl;
        std::cout << "width: " << config->width << std::endl;
        std::cout << std::endl;
      } else if (module.first.className == "HelmetModule") {
        auto config = module.second.getParams<common::WithoutHelmet>();
        if (!config) {
          std::cout << "error" << std::endl;
          return -1;
        }
        std::cout << "region: " << std::endl;
        for (auto &p : config->region) {
          std::cout << "x: " << p.at(0) << ", "
                    << "y:" << p.at(1) << std::endl;
        }

        for (auto &ap : config->algoPipelines) {
          std::cout << ap.second << std::endl;
        }

        std::cout << "threshold: " << config->threshold << std::endl;
        std::cout << std::endl;
      }
    }
  }

  gflags::ShutDownCommandLineFlags();

  return 0;
}
