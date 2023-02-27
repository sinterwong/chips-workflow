/**
 * @file configs.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "common.hpp"
#include <vector>

#ifndef _FLOWENGINE_COMMON_CONFIGS_HPP_
#define _FLOWENGINE_COMMON_CONFIGS_HPP_

namespace common {

using svector = std::vector<std::string>;

/**
 * @brief 组件的类型
 *
 */
enum class ModuleTypes { Stream, Output, Logic };

/**
 * @brief 组件的信息
 *
 */
struct ModuleInfo {
  ModuleTypes moduleType; // 组件类型
  std::string moduleName; // 组件名称
  std::string sendName;   // 下游组件
  std::string recvName;   // 上游组件
  std::string className;  // 反射类名称
};

/**
 * @brief 流组件的基本参数
 *
 */
struct StreamBase {
  int cameraId;           // 摄像机 ID
  int width;              // 视频宽度
  int height;             // 视频高度
  std::string uri;        // 流uri (file, csi, rtsp, ...)
  std::string videoCode;  // 视频编码类型（h264, h265, ..)
  std::string flowType;   // 流协议类型（rtsp, rtmp, ..)
  std::string cameraName; // 摄像机名称(uuid)
};

/**
 * @brief 输出组件基本的参数
 *
 */
struct OutputBase {
  std::string url; // 通信的url
};

/**
 * @brief 逻辑的基本参数，logic包含报警时的配置
 *
 */
struct AlarmBase {
  std::string outputDir; // 报警内容存储路径
  int videDuration;      // 报警视频录制时长
  bool isDraw;           // 报警图像是否需要标记报警信息
  float threshold; // 报警阈值（计算规则自行在不同的功能中定义）
  int eventId;      // unknow，需要原路返回给后端
  std::string page; // unknow，需要原路返回给后端
};

/**
 * @brief 前端划定区域
 *
 */
struct AttentionArea {
  std::vector<Point> region; // 划定区域
};

/**
 * @brief CV算法的基本参数
 *
 */
struct AlgoBase {
  int batchSize;         // batch of number
  svector inputNames;    // input names
  svector outputNames;   // output names
  std::string modelPath; // 算法模型
  std::string serial;    // 算法系列 {Yolo, Assd, Softmax, ...}
  Shape inputShape;      // 输入图像尺寸
  bool isScale;          // 预处理时是否等比例缩放
  float alpha;           // 预处理时除数
  float beta;            // 预处理时减数
};

/**
 * @brief 检测算法 (无所谓anchor-base和anchor-free, 后处理后都需要nms)
 *
 */
struct DetAlgo : public AlgoBase {
  float cond_thr; // 置信度阈值
  float nms_thr;  // NMS 阈值
};

/**
 * @brief 分类算法
 *
 */
struct ClassAlgo : public AlgoBase {};

// using ParamsConfig =
//     std::variant<AlgorithmConfig, CameraConfig, LogicConfig, OutputConfig>;

// inline void doSomething(AlgorithmConfig &config) {
//   // do something with AlgorithmConfig
// }

// inline void doSomething(CameraConfig &config) {
//   // do something with CameraConfig
// }

// inline void doSomething(LogicConfig &config) {
//   // do something with LogicConfig
// }

// inline void doSomething(OutputConfig &config) {
//   // do something with OutputConfig
// }

// inline void processConfig(ParamsConfig &config) {
//   std::visit([](auto &config) { doSomething(config); }, config);
// }

} // namespace common
#endif // _FLOWENGINE_COMMON_CONFIG_HPP_