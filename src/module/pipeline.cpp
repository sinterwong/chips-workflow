/**
 * @file pipelineModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-07-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "pipeline.hpp"
#include "module.hpp"
#include "statusOutputModule.h"
#include <algorithm>
#include <memory>
#include <unordered_map>

namespace module {
using common::ModuleType;

PipelineModule::PipelineModule(std::string const &config_,
                               std::string const &sendUrl_, size_t workers_n)
    : config(config_), sendUrl(sendUrl_) {
  pool = std::unique_ptr<thread_pool>(new thread_pool());
  pool->start(workers_n);
  // pool = std::unique_ptr<BS::thread_pool>(new BS::thread_pool(workers_n));
}

bool PipelineModule::submitModule(ModuleConfigure const &config,
                                  ParamsConfig const &paramsConfig) {
  switch (config.type_) {
  case ModuleType::Detection: { // Detection
    atm[config.moduleName] = std::shared_ptr<DetectModule>(
        new DetectModule(&backend, config.moduleName, "AlgorithmMessage",
                         paramsConfig.GetAlgorithmConfig(), {name}, {}));
    pool->submit(&DetectModule::go, atm.at(config.moduleName));
    break;
  }
  case ModuleType::Classifier: { // Classifier
    atm[config.moduleName] = std::shared_ptr<ClassifierModule>(
        new ClassifierModule(&backend, config.moduleName, "AlgorithmMessage",
                             paramsConfig.GetAlgorithmConfig(), {name}, {}));
    pool->submit(&ClassifierModule::go, atm.at(config.moduleName));
    break;
  }
  case ModuleType::WebStream: { // Stream
    atm[config.moduleName] = std::shared_ptr<JetsonSourceModule>(
        new JetsonSourceModule(&backend, config.moduleName, "FrameMessage",
                               paramsConfig.GetCameraConfig(), {name}, {}));
    pool->submit(&JetsonSourceModule::go, atm.at(config.moduleName));
    break;
  }
  case ModuleType::AlarmOutput: { // Output
    atm[config.moduleName] = std::shared_ptr<AlarmOutputModule>(
        new AlarmOutputModule(&backend, config.moduleName, "OutputMessage",
                             paramsConfig.GetOutputConfig(), {name}, {}));
    pool->submit(&AlarmOutputModule::go, atm.at(config.moduleName));
    break;
  }
  case ModuleType::StatusOutput: { // Output
    atm[config.moduleName] = std::shared_ptr<StatusOutputModule>(
        new StatusOutputModule(&backend, config.moduleName, "OutputMessage",
                             paramsConfig.GetOutputConfig(), {name}, {}));
    pool->submit(&AlarmOutputModule::go, atm.at(config.moduleName));
    break;
  }
  case ModuleType::Calling: { // Calling
    atm[config.moduleName] = std::shared_ptr<CallingModule>(
        new CallingModule(&backend, config.moduleName, "LogicMessage",
                          paramsConfig.GetLogicConfig(), {name}, {}));
    pool->submit(&CallingModule::go, atm.at(config.moduleName));
    break;
  }
  default: {
    break;
  }
  }
  return true;
}

bool PipelineModule::parseConfigs(
    std::string const &path,
    std::vector<std::vector<std::pair<ModuleConfigure, ParamsConfig>>>
        &pipelines) {
  std::string content;
  if (!configParser.readFile(path, content)) {
    FLOWENGINE_LOGGER_ERROR("config parse: read file is failed!");
    return false;
  }

  if (!configParser.parseConfig(content.c_str(), pipelines)) {
    FLOWENGINE_LOGGER_INFO("config parse: parseParams is failed!");
    return false;
  }
  return true;
}

void PipelineModule::attachModule(std::string const &moduleName,
                                  std::string const &sendModule,
                                  std::string const &recvModule) {
  if (!sendModule.empty()) {
    atm.at(moduleName)->addSendModule(sendModule);
    atm.at(sendModule)->addRecvModule(moduleName);
  }

  if (!recvModule.empty()) {
    atm.at(moduleName)->addRecvModule(recvModule);
    atm.at(recvModule)->addSendModule(moduleName);
  }
}

void PipelineModule::detachModule(std::string const &moduleName) {
  // 找到并解除输入模块的所有关联
  auto iter = atm.find(moduleName);
  if (iter == atm.end()) {
    FLOWENGINE_LOGGER_ERROR("{} is not runing", moduleName);
    return;
  }
  for (auto &sm : atm.at(moduleName)->getSendModule()) {
    atm.at(sm)->delRecvModule(moduleName);
  };

  for (auto &rm : atm.at(moduleName)->getRecvModule()) {
    if (rm == name) { // 保留pipeline控制模块
      continue;
    }
    atm.at(rm)->delSendModule(moduleName);
  };
}

void PipelineModule::stopModule(std::string const &moduleName) {

  // 解除关联
  detachModule(moduleName);

  // 发送终止正在 go 的消息
  backend.message->send(name, moduleName, type, queueMessage());

  // 从atm中删除对象
  int num = atm.erase(moduleName);
}

bool PipelineModule::startPipeline() {
  std::vector<std::vector<std::pair<ModuleConfigure, ParamsConfig>>> pipelines;
  if (!parseConfigs(config, pipelines)) {
    FLOWENGINE_LOGGER_ERROR("parse config error");
    return false;
  }
  std::vector<std::string> currentModules;
  std::vector<ModuleConfigure> moduleRelations;
  // run 起来所有模块并且制作所有需要关联的模块, 后续可能会有所扩展
  for (auto &pipeline : pipelines) {
    std::string streamName;
    for (auto &config : pipeline) {
      currentModules.push_back(config.first.moduleName);
      if (config.first.ctype == "stream") {
        streamName = config.first.moduleName;
      }
      moduleRelations.push_back(config.first);
      if (atm.find(config.first.moduleName) == atm.end()) {
        submitModule(config.first, config.second);
        // 逻辑模块因为需要写出视频, 因此也需要关联上stream
        if (config.first.ctype == "logic") {
          ModuleConfigure mc{config.first};
          mc.sendName = "";
          mc.recvName = streamName;
          moduleRelations.emplace_back(mc);
        }
      }
    }
  }
  // 清掉已经停用的模块
  if (!currentModules.empty()) {
    std::unordered_map<std::string, std::shared_ptr<Module>>::iterator iter;
    for (iter = atm.begin(); iter != atm.end();) {
      auto it =
          std::find(currentModules.begin(), currentModules.end(), iter->first);
      if (it == currentModules.end()) {
        // 说明该模块需要关闭（函数中存在删除atm的部分）
        // 解除关联
        detachModule(iter->first);
        // 发送终止正在 go 的消息
        backend.message->send(name, iter->first, type, queueMessage());
        // 从atm中删除对象
        iter = atm.erase(iter);
      } else {
        ++iter;
      }
    }

    // 关联模块
    for (auto &mc : moduleRelations) {
      attachModule(mc.moduleName, mc.sendName, mc.recvName);
    }
  }

  return true;
}

void PipelineModule::go() {
  while (true) {
    startPipeline();
    std::this_thread::sleep_for(std::chrono::seconds(15));
    // terminate();  // 终止所有任务
    // break;
  }
}

} // namespace module
