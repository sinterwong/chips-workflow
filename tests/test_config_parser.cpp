#include "gflags/gflags.h"
#include "module/configParser.hpp"
DEFINE_string(config_path, "", "Specify the path of image.");

using namespace module::utils;

using common::ClassAlgo;
using common::DetAlgo;

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ConfigParser parser;
  std::vector<PipelineParams> pipelines;
  std::vector<AlgorithmParams> algorithms;

  parser.parseConfig(FLAGS_config_path, pipelines, algorithms);

  for (auto &algo : algorithms) {
    std::cout << algo.first << std::endl;
    algo.second.visitParams([](auto &&params) {
      // 对于不同类型的参数，可以根据其类型进行不同的处理
      using T = std::decay_t<decltype(params)>; // 获取参数的实际类型
      if constexpr (std::is_same_v<T, DetAlgo>) {
        std::cout << "DetAlgo: " << params.serial << "\n";
      } else if constexpr (std::is_same_v<T, ClassAlgo>) {
        std::cout << "ClassAlgo: " << params.serial << "\n";
      } else {
        std::cout << "Unknown params\n";
      }
    });
  }

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
        for (auto &region : config->regions) {
          for (auto &p : region) {
            std::cout << "x: " << p.at(0) << ", "
                      << "y:" << p.at(1) << std::endl;
          }
        }

        for (auto &ap : config->algoPipelines) {
          std::cout << ap.first << ", " << ap.second.attentions.at(0)
                    << std::endl;
        }

        std::cout << "threshold: " << config->threshold << std::endl;
        std::cout << std::endl;
      }
    }
  }

  gflags::ShutDownCommandLineFlags();

  return 0;
}
