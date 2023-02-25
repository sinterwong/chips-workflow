/**
 * @file messageBus.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-30
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef DETTRACKENGINE_MESSAGEBUS_H
#define DETTRACKENGINE_MESSAGEBUS_H

#include "common/common.hpp"
#include <array>
#include <atomic>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

using common::ColorType;
using common::RetBox;
using common::RetPoly;

/**
 * @brief 报警时需要返回的信息
 *
 */
struct AlarmInfo {
  int height;                  // 高
  int width;                   // 宽
  int cameraId;                // 摄像机 ID
  int eventId;                 // 报警类型Id
  int prepareDelayInSec;       // 视频录制时间
  std::string page;            // 是否展示报警
  std::string cameraIp;        // 视频流 IP
  std::string alarmType;       // 报警类型
  std::string alarmFile;       // 报警图片(base64)
  std::string alarmId;         // 本次报警唯一 ID
  std::string alarmDetails;    // 报警细节
  std::string algorithmResult; // 算法返回结果
};

/**
 * @brief 模块之间的传递的消息
 *
 */
struct queueMessage {
  int key;          // 帧id
  int status;       // 上游状态
  std::string send; // 上游模块名称
  std::string recv;
  std::string messageType;
  ColorType frameType;
  AlarmInfo alarmInfo;
};

/**
 * @brief 模块控制中心
 *
 */
class MessageBus {
public:
  enum class returnFlag {
    null,
    mapNotFind,
    successWithEmpty,
    successWithMore,
  };

  virtual bool registered(std::string name) = 0;

  virtual bool send(std::string source, std::string target, std::string type,
                    queueMessage message) = 0;

  virtual bool recv(std::string source, returnFlag &flag, std::string &send,
                    std::string &type, queueMessage &byte,
                    bool waitFlag = true) = 0;
};

#endif // DETTRACKENGINE_MESSAGEBUS_H
