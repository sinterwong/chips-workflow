/**
 * @file CallingModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-07-31
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "callingModule.h"

namespace module {

CallingModule::CallingModule(Backend *ptr, const std::string &initName,
                             const std::string &initType,
                             const common::LogicConfig &logicConfig,
                             const std::vector<std::string> &recv,
                             const std::vector<std::string> &send,
                             const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool) {
}

void CallingModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  if (recvModule.empty()) {
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      FLOWENGINE_LOGGER_INFO("{} CallingModule module was done!", name);
      stopFlag.store(true);
    } else if (type == "AlgorithmMessage") {
      // 此处根据 buf.algorithmResult 写吸烟的逻辑并填充 buf.alarmResult 信息
      // 如果符合条件就发送至SendOutputModule
      // std::cout << "CallingModule: Upstream bbox size: " << buf.algorithmResult.bboxes.size() << std::endl;
      for (auto &bbox: buf.algorithmResult.bboxes) {
        // std::cout << "CallingModule: Upstream bbox name: " << bbox.first << std::endl;
      }
      // std::cout << "*********************" << std::endl;
    }
    autoSend(buf);
  }
}
} // namespace module
