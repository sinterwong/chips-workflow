/**
 * @file module_header.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "infer_header.hpp"
#include <chrono>

#ifndef _FLOWENGINE_COMMON_MODULE_HEADER_HPP_
#define _FLOWENGINE_COMMON_MODULE_HEADER_HPP_

namespace common {

using namespace std::chrono_literals;

/**
 * @brief 组件之间发送消息的类型
 *
 */
enum class MessageType { None, Stream, Status, Close };

/**
 * @brief 组件的信息
 *
 */
struct ModuleInfo {
  std::string moduleName; // 组件名称
  std::string moduleType; // 组件类型
  std::string sendName;   // 下游组件
  std::string recvName;   // 上游组件
  std::string className;  // 反射类名称

  friend std::ostream &operator<<(std::ostream &os, ModuleInfo &param) {
    os << "moduleName: " << param.moduleName << "\n"
       << "moduleType: " << param.moduleType << "\n"
       << "sendName: " << param.sendName << "\n"
       << "recvName: " << param.recvName << "\n"
       << "className: " << param.className << std::endl;
    return os;
  }
};

/**
 * @brief module 基本参数
 *
 */
struct ModuleBase {
  int64_t interval = 100; // 模块执行后间隔时间(ms)
};

/**
 * @brief 流组件的基本参数
 *
 */
struct StreamBase : public ModuleBase {
  int cameraId;           // 摄像机 ID
  int width;              // 视频宽度
  int height;             // 视频高度
  std::string uri;        // 流uri (file, csi, rtsp, ...)
  std::string videoCode;  // 视频编码类型（h264, h265, ..)
  std::string flowType;   // 流协议类型（rtsp, rtmp, ..)
  std::string cameraName; // 摄像机名称(uuid)
  int runTime;            // 每次轮询执行多长时间
};

/**
 * @brief 输出组件基本的参数
 *
 */
struct OutputBase : public ModuleBase {
  std::string url; // 通信的url
};

/**
 * @brief 针对算法结果处理方式的特定参数
 *
 */
struct AlgoParams {
  std::vector<int> attentions;         // 算法需要保留的类别
  std::vector<std::string> basedNames; // 算法基于谁的结果
  float cropScaling;                   // 抠图需要放缩的比例
};

// 功能模块中需要启动使用的算法
using algo_pipelines = std::vector<std::pair<std::string, AlgoParams>>;

/**
 * @brief 逻辑的基本参数，logic包含报警时的配置
 *
 */
struct LogicBase : public ModuleBase {
  algo_pipelines algoPipelines;  // 算法执行的pipeline
  std::vector<Points2i> regions; // 划定区域
  int eventId;                   // unknow，需要原路返回给后端
  std::string page;              // unknow，需要原路返回给后端
  std::string outputDir;         // 报警内容存储路径
  int videoDuration;             // 报警视频录制时长
  uint8_t drawType;              // 报警图像是否需要标记报警信息
};

/**
 * @brief OCR类型的算法逻辑
 *
 */
struct OCRConfig : public LogicBase {
  OCRConfig(LogicBase &&alarm, std::string &&chars_)
      : LogicBase(alarm), chars(chars_) {}

  std::string chars; // 需要匹配的字符集
};

/**
 * @brief 计数监控
 *
 */
struct ObjectCounterConfig : public LogicBase {
  ObjectCounterConfig(LogicBase &&alarm, int amount_)
      : LogicBase(alarm), amount(amount_) {}

  int amount; // 达到一定数量之后报警
};

/**
 * @brief 依靠检测与分类类型的报警模块
 *
 */
struct DetClsMonitor : public LogicBase {
  DetClsMonitor(LogicBase &&logic_, float threshold_,
                size_t requireExistence_ = 1)
      : LogicBase(logic_), threshold(threshold_),
        requireExistence(requireExistence_) {}

  float threshold;         // 报警阈值
  size_t requireExistence; // 要求目标类别存在或不存在
};

/**
 * @brief module 参数中心
 *
 */
class ModuleConfig {
public:
  // 将所有参数类型存储在一个 std::variant 中
  using Params = std::variant<StreamBase, OutputBase, OCRConfig, DetClsMonitor,
                              ObjectCounterConfig>;

  // 设置参数
  template <typename T> void setParams(T params) {
    params_ = std::move(params);
  }

  // 访问参数
  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               params_);
  }

  // 获取参数
  template <typename T> T *getParams() { return std::get_if<T>(&params_); }

private:
  Params params_;
};

} // namespace common
#endif // _FLOWENGINE_COMMON_CONFIG_HPP_