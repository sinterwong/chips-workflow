/**
 * @file common.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-04-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <array>
#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

#ifndef _FLOWENGINE_COMMON_COMMON_HPP_
#define _FLOWENGINE_COMMON_COMMON_HPP_

namespace common {

enum class WORKER_TYPES { SMOKE, PHONE, FD};

struct ParamsConfig {
  std::array<int, 4> region;
  float cond_thr = 0.3;                       // 置信度阈值
  float nms_thr = 0.5;                        // NMS 阈值
  std::string modelDir;                       // engine 所在目录
  std::array<int, 3> originShape;             // 原始的输入尺度
};

// 接收管理进程给出发生变化的配置
struct FlowConfigure {
  bool status;  // 只有启动和关闭两种状态 true 启动，false 关闭
  int cameraId; // 摄像机唯一ID
  int hostId;   // 设备主机id
  int provinceId; // 省
  int cityId;     // 市
  int regionId;   // 区
  int stationId;  // 站
  int width;      // 视频宽度
  int height;     // 视频高度
  int location;   // 站内的那个区域（卸油区、加油区等等）
  std::string alarmType;     // 报警类型（smoke, phone等等）
  std::string videoCode;     // 视频编码类型
  std::string flowType;      // 流协议类型
  std::string cameraIp;      // 网络流链接
  ParamsConfig paramsConfig; // 算法配置参数
};

struct AlarmInfo {
  int cameraId;   // 摄像机唯一ID
  int hostId;     // 设备主机id
  int provinceId; // 省
  int cityId;     // 市
  int regionId;   // 区
  int stationId;  // 站
  int width;      // 视频宽度
  int height;     // 视频高度
  int location;   // 站内的那个区域（卸油区、加油区等等）
  std::string alarmType; // 报警类型（smoke, phone等等）
  std::string alarmFile; // 报警图片
  std::string alarmId;   // 报警的唯一标识 uuid
  std::string alarmDetails; // 报警细节（我也不知道是啥，就给报警类型就行）
  std::string cameraIp;  // 网络流链接
  std::string resultInfo; // 本次执行的所有算法结果信息
};

} // namespace common
#endif
