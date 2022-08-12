/**
 * @file alarmOutputModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_SEND_ALARM_OUTPUT_H_
#define __METAENGINE_SEND_ALARM_OUTPUT_H_

#include <any>
#include <curl/curl.h>
#include <memory>
#include <vector>

#include "messageBus.h"
#include "outputModule.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "outputModule.h"

namespace module {

struct AlarmInfo {
  int height;                  // 高
  int width;                   // 宽
  int cameraId;                // 摄像机 ID
  std::string cameraIp;        // 视频流 IP
  std::string alarmType;       // 报警类型
  std::string alarmFile;       // 报警图片(base64)
  std::string alarmId;         // 本次报警唯一 ID
  std::string alarmDetails;    // 报警细节
  std::string algorithmResult; // 算法返回结果
};

class AlarmOutputModule : public OutputModule {

public:
  AlarmOutputModule(Backend *ptr, const std::string &initName,
                   const std::string &initType,
                   const common::OutputConfig &outputConfig,
                   const std::vector<std::string> &recv = {},
                   const std::vector<std::string> &send = {},
                   const std::vector<std::string> &pool = {});
  ~AlarmOutputModule() {}

  void forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
                   message) override;

  bool postResult(std::string const &url, AlarmInfo const &resultInfo,
                  std::string &result);

  bool writeResult(AlgorithmResult const &rm, std::string &result);


};
} // namespace module
#endif // __METAENGINE_SEND_ALARM_OUTPUT_H_
