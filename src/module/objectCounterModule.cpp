/**
 * @file objectCounterModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-04-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "objectCounterModule.h"
#include "logger/logger.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace module {

/**
 * @brief
 *
 * @param message
 */
void ObjectCounterModule::forward(std::vector<forwardMessage> &message) {
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

    // 各个算法结果的区域
    std::unordered_map<std::string, std::vector<common::RetBox>> algoRegions;

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
    algoRegions["regions"] = std::move(regions);

    // 根据提供的配置执行算法，
    // auto &apipes = config->algoPipelines;
    
  }
}

FlowEngineModuleRegister(ObjectCounterModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
