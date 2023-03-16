#include "gflags/gflags.h"
#include "common/common.hpp"

DEFINE_string(image_path, "", "Specify the path of image.");

using namespace common;

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ModuleConfig center;

  // 创建SmokingMonitor
  AttentionArea area;
  LogicBase alarm;
  InferInterval inferInterval;

  DetClsMonitor smoking{std::move(area), std::move(alarm),
                         std::move(inferInterval)};

  center.setParams(std::move(smoking));
  // 访问参数
  center.visitParams([](auto &&params) {
    // 对于不同类型的参数，可以根据其类型进行不同的处理
    using T = std::decay_t<decltype(params)>; // 获取参数的实际类型
    if constexpr (std::is_same_v<T, StreamBase>) {
      std::cout << "StreamBase: " << params.uri << "\n";
    } else if constexpr (std::is_same_v<T, DetClsMonitor>) {
      std::cout << "DetClsMonitor: " << params.interval.count() << "\n";
    } else if constexpr (std::is_same_v<T, WithoutHelmet>) {
      std::cout << "WithoutHelmetMonitor: " << params.outputDir << "\n";
    } else {
      std::cout << "Unknown params\n";
    }
  });
  return 0;
}