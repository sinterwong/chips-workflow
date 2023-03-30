/**
 * @file ocrModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-03-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "ocrModule.h"
#include "logger/logger.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>

using json = nlohmann::json;

namespace module {

/**
 * @brief
 *
 * @param message
 */
void CharsRecognitionModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} HelmetModule module was done!", name);
      stopFlag.store(true);
      return;
    }

    // 读取图片
    FrameBuf frameBufMessage = ptr->pool->read(buf.key);
    auto image =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

    // 初始待计算区域，每次算法结果出来之后需要更新regions
    std::vector<common::RetBox> regions;
    for (auto const &area : config->regions) {
      regions.emplace_back(common::RetBox{
          name,
          {static_cast<float>(area[0][0]), static_cast<float>(area[0][1]),
           static_cast<float>(area[1][0]), static_cast<float>(area[1][1]), 0.0,
           0.0}});
    }
    if (regions.empty()) {
      // 前端没有画框
      regions.emplace_back(common::RetBox{name, {0, 0, 0, 0, 0, 0}});
    }

    // 至此，所有的算法模块执行完成，整合算法结果判断是否报警

    // 本轮算法结果生成
    json algoRet;
    buf.alarmInfo.algorithmResult = algoRet.dump();
    autoSend(buf);
  }
}

FlowEngineModuleRegister(CharsRecognitionModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
