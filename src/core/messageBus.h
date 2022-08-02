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
  std::string cameraId;  // 摄像机唯一ID
  std::string videoCode; // 视频编码类型
  std::string flowType;  // 流协议类型
  std::string cameraIp;  // 网络流链接
  int provinceId;        // 省ID
  int cityId;            // 市ID
  int regionId;          // 区ID
  int stationId;         // 站ID
  int location;          // 所在位置（卸油区，加油区...）
};

/**
 * @brief 报警信息
 * 
 */
struct AlarmResult {
  std::string cameraId;     // 摄像机唯一ID
  std::string alarmType;    // 报警类型（smoke, phone等等）
  std::string alarmFile;    // 报警图片
  std::string alarmId;      // 报警的唯一标识 uuid
  std::string alarmDetails; // 报警细节
  std::string cameraIp;     // 网络流链接
};

/**
 * @brief 报警时的算法信息
 * 
 */
struct AlgorithmResult {
  std::vector<std::pair<std::string, std::array<float, 6>>>
      bboxes; // [x1, y1, x2, y2, confidence, classid]
  std::vector<std::pair<std::string, std::array<float, 9>>>
      polys; // [x1, y1, ..., x4, y4, classid]
};

struct queueMessage {
  int width;
  int height;
  int key;
  std::string send;
  std::string recv;
  std::string messageType;
  std::string frameType;
  std::string str;
  AlarmResult alarmResult;
  CameraResult cameraResult;
  AlgorithmResult algorithmResult;
};

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
