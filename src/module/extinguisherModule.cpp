/**
 * @file extinguisherModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-23
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "extinguisherModule.h"
#include <cstdlib>
#include <experimental/filesystem>
#include <sys/stat.h>
#include <unistd.h>
namespace module {

ExtinguisherModule::ExtinguisherModule(Backend *ptr,
                                       const std::string &initName,
                                       const std::string &initType,
                                       const common::LogicConfig &logicConfig)
    : LogicModule(ptr, initName, initType, logicConfig) {}

/**
 * @brief
 * 1. recv 类型：logic, algorithm
 * 2. send 类型：algorithm, logic
 *
 * @param message
 */
void ExtinguisherModule::forward(std::vector<forwardMessage> &message) {
  if (recvModule.empty()) {
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      FLOWENGINE_LOGGER_INFO("{} ExtinguisherModule module was done!", name);
      stopFlag.store(true);
      return;
    }

    if (isRecord) {
      if (type == "stream") {
        recordVideo(buf.key, buf.cameraResult.widthPixel,
                    buf.cameraResult.heightPixel);
      }
      continue;
    }

    if (type == "algorithm") {
      // 此处根据 buf.algorithmResult 写吸烟的逻辑并填充 buf.alarmResult 信息
      // 如果符合条件就发送至AlarmOutputModule
      for (int i = 0; i < static_cast<int>(buf.algorithmResult.bboxes.size()); i++) {
        auto &bbox = buf.algorithmResult.bboxes.at(i);
        if (bbox.first != send) {
          continue;
        }
        FLOWENGINE_LOGGER_CRITICAL("classid: {}, confidence: {}",
                                   bbox.second.at(5), bbox.second.at(4));
        // if (bbox.second.at(5) == 0 && bbox.second.at(4) > 0.9) {
        if (bbox.second.at(5) == 0) {
          // 生成报警信息和报警图
          generateAlarm(buf, "未检测到灭火器", bbox);

          // 发送至后端
          sendWithTypes(buf, {"output"});

          // 保存视频
          if (params.videDuration > 0) {
            initRecord(buf);
          }
        }
        break;
      }
    } else if (type == "stream") {
      // 配置算法推理时需要用到的信息
      buf.logicInfo.region = params.region;
      buf.logicInfo.attentionClasses = params.attentionClasses;
      // 不能发送给output
      sendWithTypes(buf, {"algorithm"});
    }
  }
}

FlowEngineModuleRegister(ExtinguisherModule, Backend *, std::string const &,
                         std::string const &, common::LogicConfig const &);
} // namespace module
