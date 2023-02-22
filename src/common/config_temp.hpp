/**
 * @file config.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#ifndef _FLOWENGINE_COMMON_CONFIG_HPP_
#define _FLOWENGINE_COMMON_CONFIG_HPP_

namespace common {

enum class ConfigType { Algorithm, Stream, Output, Logic, None };

struct ModuleConfigure {
  std::string typeName;
  std::string ctype;
  std::string moduleName; // 模块名称
  std::string sendName;   // 下游模块
  std::string recvName;   // 上游模块
};

struct AlgorithmConfig {
  std::string modelPath;                      // engine 所在目录
  std::vector<std::string> inputTensorNames;  // input tensor names
  std::vector<std::string> outputTensorNames; // output tensor names
  std::string algorithmSerial;                // 算法系列
  Shape inputShape;              // 算法需要的输入尺度
  bool isScale;                               // 是否等比例缩放
  float cond_thr;                             // 置信度阈值
  float nms_thr;                              // NMS 阈值
  float alpha;                                // 预处理时除数
  float beta;                                 // 预处理时减数
  int batchSize;                              // batch of number
};

struct CameraConfig {
  int widthPixel;         // 视频宽度
  int heightPixel;        // 视频高度
  int cameraId;           // 摄像机 ID
  std::string cameraName; // 摄像机名称(uuid)
  std::string videoCode;  // 视频编码类型
  std::string flowType;   // 流协议类型
  std::string cameraIp;   // 网络流链接
};

struct LogicConfig {
  std::string outputDir;
  int eventId;
  int videDuration;
  std::string page;
  std::array<int, 4> region;         // 划定区域
  std::vector<int> attentionClasses; // 划定区域
  float threshold;
  bool isDraw = true;
};

struct OutputConfig {
  std::string url;
};

using ParamsConfig = std::variant<AlgorithmConfig, CameraConfig, LogicConfig, OutputConfig>;

inline void doSomething(AlgorithmConfig &config) {
  // do something with AlgorithmConfig
}

inline void doSomething(CameraConfig &config) {
  // do something with CameraConfig
}

inline void doSomething(LogicConfig &config) {
  // do something with LogicConfig
}

inline void doSomething(OutputConfig &config) {
  // do something with OutputConfig
}

inline void processConfig(ParamsConfig &config) {
  std::visit([](auto &config) { doSomething(config); }, config);
}

}
#endif // _FLOWENGINE_COMMON_CONFIG_HPP_