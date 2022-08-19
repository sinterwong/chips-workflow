//
// Created by Wallel on 2021/12/26.
//

#ifndef DETTRACKENGINE_MESSAGEBUS_H
#define DETTRACKENGINE_MESSAGEBUS_H

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

// #include <boost/pool/object_pool.hpp>
// #include <boost/pool/pool.hpp>

// #include "basicMessage.pb.h"

/**
 * @brief 报警时的摄像头信息
 *
 */
struct CameraResult {
  int widthPixel;        // 视频宽度
  int heightPixel;       // 视频高度
  int cameraId;          // 摄像机唯一ID
  std::string videoCode; // 视频编码类型
  std::string flowType;  // 流协议类型
  std::string cameraIp;  // 网络流链接
};

/**
 * @brief 报警信息
 *
 */
struct AlarmResult {
  int alarmVideoDuration;   // 报警视频时长（秒）
  int eventId;              // 报警类型的ID（smoke, phone等等）
  std::string page;         // 是否展示报警
  std::string alarmType;    // 报警类型（smoke, phone等等）
  std::string alarmFile;    // 报警图片路径
  std::string alarmId;      // 报警的唯一标识 uuid
  std::string alarmDetails; // 报警细节
};

/**
 * @brief 报警时的算法信息
 * @todo 这里的vector线程不安全 未来需要替换
 */
struct AlgorithmResult {
  std::vector<std::pair<std::string, std::array<float, 6>>>
      bboxes; // [x1, y1, x2, y2, confidence, classid]
  std::vector<std::pair<std::string, std::array<float, 9>>>
      polys; // [x1, y1, ..., x4, y4, classid]
};

/**
 * @brief 逻辑模块配置
 */
struct LogicInfo {
  std::array<int, 4> region; // [x1, y1, x2, y2]
  std::vector<int> attentionClasses; // [0, 2, 10...]
};

/**
 * @brief 传递的消息
 *
 */
struct queueMessage {
  int width;        // 帧宽
  int height;       // 帧高
  int key;          // 帧id
  int status;       // 上游状态
  std::string send; // 上游模块名称
  std::string recv;  
  std::string messageType;
  std::string frameType;
  LogicInfo logicInfo;
  AlarmResult alarmResult;
  CameraResult cameraResult;
  AlgorithmResult algorithmResult;
};

/**
 * @brief 模块控制中心
 *
 */
class MessageBus {
protected:
  std::unordered_set<std::string> pool;

public:
  enum returnFlag {
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
