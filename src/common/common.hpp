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

#include "config.hpp"

#ifndef _FLOWENGINE_COMMON_COMMON_HPP_
#define _FLOWENGINE_COMMON_COMMON_HPP_

namespace common {

struct AlarmInfo {
  int hostId;     // 设备主机id
  int provinceId; // 省
  int cityId;     // 市
  int regionId;   // 区
  int stationId;  // 站
  int width;      // 视频宽度
  int height;     // 视频高度
  int location;   // 站内的那个区域（卸油区、加油区等等）
  std::string cameraId;  // 摄像机唯一ID
  std::string alarmType; // 报警类型（smoke, phone等等）
  std::string alarmFile; // 报警图片
  std::string alarmId;   // 报警的唯一标识 uuid
  std::string alarmDetails; // 报警细节（我也不知道是啥，就给报警类型就行）
  std::string cameraIp;   // 网络流链接
  std::string resultInfo; // 本次执行的所有算法结果信息
};

} // namespace common
#endif
