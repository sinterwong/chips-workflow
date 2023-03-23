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
using svector = std::vector<std::string>;

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
struct LogicBase {
  std::string outputDir; // 报警内容存储路径
  int videDuration;      // 报警视频录制时长
  bool isDraw;           // 报警图像是否需要标记报警信息
  float threshold; // 报警阈值（计算规则自行在不同的功能中定义）
  algo_pipelines algoPipelines;
  int eventId;      // unknow，需要原路返回给后端
  std::string page; // unknow，需要原路返回给后端
};

/**
 * @brief 前端划定区域
 *
 */
struct AttentionArea {
  std::vector<Points2i> regions; // 划定区域
};

/**
 * @brief 每次执行模块的间隔时间
 *
 */
struct InferInterval {
  std::chrono::seconds interval{3}; // 间隔时间
};

/**
 * @brief 未佩戴安全帽识别
 *
 */
struct WithoutHelmet : public AttentionArea,
                       public LogicBase,
                       public InferInterval {
  WithoutHelmet(AttentionArea &&aaera, LogicBase &&alarm,
                InferInterval &&interval)
      : AttentionArea(aaera), LogicBase(alarm), InferInterval(interval) {}
};

/**
 * @brief 通用模块
 *
 */
struct DetClsMonitor : public AttentionArea,
                       public LogicBase,
                       public InferInterval {
  DetClsMonitor(AttentionArea &&aaera, LogicBase &&alarm,
                InferInterval &&interval)
      : AttentionArea(aaera), LogicBase(alarm), InferInterval(interval) {}
};

/**
 * @brief module 参数中心
 *
 */
class ModuleConfig {
public:
  // 将所有参数类型存储在一个 std::variant 中
  using Params = std::variant<StreamBase, OutputBase, LogicBase, WithoutHelmet,
                              DetClsMonitor>;

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