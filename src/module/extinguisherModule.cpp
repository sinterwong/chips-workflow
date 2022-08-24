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

ExtinguisherModule::ExtinguisherModule(Backend *ptr, const std::string &initName,
                             const std::string &initType,
                             const common::LogicConfig &logicConfig,
                             const std::vector<std::string> &recv,
                             const std::vector<std::string> &send,
                             const std::vector<std::string> &pool)
    : LogicModule(ptr, initName, initType, logicConfig, recv, send, pool) {}

/**
 * @brief 
 * 1. recv 类型：logic, algorithm
 * 2. send 类型：algorithm, logic
 * 
 * @param message 
 */
void ExtinguisherModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  if (recvModule.empty()) {
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      // FLOWENGINE_LOGGER_INFO("{} ExtinguisherModule module was done!", name);
      std::cout << name << "{} ExtinguisherModule module was done!" << std::endl;
      stopFlag.store(true);
      return;
    }

    if (type == "algorithm") {
      // 此处根据 buf.algorithmResult 信息判断是否存在灭火器
      // 如果符合条件就发送至AlarmOutputModule
      for (int i = 0; i < buf.algorithmResult.bboxes.size(); i++) {
        auto &bbox = buf.algorithmResult.bboxes.at(i);
        if (bbox.first == send) {
          // std::cout << "classid: " << bbox.second.at(5) << ", "
          //           << "confidence: " << bbox.second.at(4) << std::endl;
          if (bbox.second.at(5) == 1 && bbox.second.at(4) > 0.93) { // 存在报警
            // 生成本次报警的唯一ID
            buf.alarmResult.alarmDetails = "不存在灭火器";

            // 记录当前的框为报警框
            std::pair<std::string, std::array<float, 6>> b{bbox};
            b.first = name;
            // 单独画出报警框
            buf.algorithmResult.bboxes.emplace_back(b);
            // 仅发送回父级逻辑或输出
            sendWithTypes(buf, {"logic"});
            break;
          }
        }
      }
    } else if (type == "logic") {
      // 父级逻辑调用
      buf.logicInfo.region = params.region;
      buf.logicInfo.attentionClasses = params.attentionClasses;
      // 只发送给算法
      sendWithTypes(buf, {"algorithm"});
    }
  }
}

FlowEngineModuleRegister(ExtinguisherModule, Backend *, std::string const &,
                         std::string const &, common::LogicConfig const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &);
} // namespace module
