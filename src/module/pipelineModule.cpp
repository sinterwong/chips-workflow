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

#include "pipelineModule.hpp"
#include "callingModule.h"
#include "jetson/classifierModule.h"
#include "jetson/jetsonSourceModule.h"
#include "logger/logger.hpp"
#include "sendOutputModule.h"
#include <chrono>
#include <memory>
#include <utility>
#include <vector>

namespace module {
using common::ModuleType;

PipelineModule::PipelineModule(std::string const &sendUrl_, size_t workers_n)
    : sendUrl(sendUrl_) {
  pool = std::unique_ptr<thread_pool>(new thread_pool());
  pool->start(workers_n);

  // 启动SendOutput模块，此模块一个进程只此一个
  ModuleConfigure config{ModuleType::Output, "Output", "", ""};
  common::OutputConfig outputConfig = {sendUrl};
  ParamsConfig paramsConfig{outputConfig};
  submitModule(config, paramsConfig);
}

bool PipelineModule::submitModule(ModuleConfigure const &config,
                                  ParamsConfig const &paramsConfig) {
  switch (config.type_) {
  case ModuleType::Detection: { // Detection
    atm[config.moduleName] = std::shared_ptr<DetectModule>(
        new DetectModule(&backend, config.moduleName, "AlgorithmMessage",
                         paramsConfig.GetAlgorithmConfig(), {name}, {}));
    pool->submit(&DetectModule::go, atm[config.moduleName]);
    break;
  }
  case ModuleType::Classifier: { // Classifier
    atm[config.moduleName] = std::shared_ptr<ClassifierModule>(
        new ClassifierModule(&backend, config.moduleName, "AlgorithmMessage",
                             paramsConfig.GetAlgorithmConfig(), {name}, {}));
    pool->submit(&ClassifierModule::go, atm[config.moduleName]);
    break;
  }
  case ModuleType::Stream: { // Stream
    atm[config.moduleName] = std::shared_ptr<JetsonSourceModule>(
        new JetsonSourceModule(&backend, config.moduleName, "FrameMessage",
                               paramsConfig.GetCameraConfig(), {}, {}));
    pool->submit(&JetsonSourceModule::go, atm[config.moduleName]);
    break;
  }
  case ModuleType::Output: { // Output
    atm[config.moduleName] = std::shared_ptr<SendOutputModule>(
        new SendOutputModule(&backend, config.moduleName, "OutputMessage",
                             paramsConfig.GetOutputConfig(), {name}, {}));
    pool->submit(&SendOutputModule::go, atm[config.moduleName]);
    break;
  }
  case ModuleType::Calling: { // Calling
    atm[config.moduleName] = std::shared_ptr<CallingModule>(
        new CallingModule(&backend, config.moduleName, "LogicMessage",
                          paramsConfig.GetLogicConfig(), {name}, {}));
    pool->submit(&CallingModule::go, atm[config.moduleName]);
    break;
  }
  default: {
    break;
  }
  }
  addModule(config.moduleName, config.sendName, config.recvName);
  return true;
}

bool PipelineModule::parseConfigs(
    std::string const &path,
    std::vector<std::vector<std::pair<ModuleConfigure, ParamsConfig>>> &pipelines) {
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

void PipelineModule::addModule(std::string const &moduleName,
                               std::string const &sendModule,
                               std::string const &recvModule) {
  if (!sendModule.empty()) {
    auto iter = std::find(atm[moduleName]->sendModule.begin(),
                          atm[moduleName]->sendModule.end(), sendModule);
    if (iter == atm[moduleName]->sendModule.end()) {
      atm[moduleName]->addSendModule(sendModule);
    }
  }

  if (!recvModule.empty()) {
    auto iter = std::find(atm[moduleName]->recvModule.begin(),
                          atm[moduleName]->recvModule.end(), recvModule);
    if (iter == atm[moduleName]->recvModule.end()) {
      atm[moduleName]->addRecvModule(recvModule);
    }
  }
}

bool PipelineModule::startPipeline(std::string const &uri) {
  std::vector<
      std::vector<std::pair<ModuleConfigure, ParamsConfig>>> pipelines;
  if (!parseConfigs(uri, pipelines)) {
    FLOWENGINE_LOGGER_ERROR("parse config error");
    return false;
  }
  for (auto &pipeline : pipelines) {
    for (auto &config : pipeline) {
      if (atm.find(config.first.moduleName) == atm.end()) {
        submitModule(config.first, config.second);
      } else {
        addModule(config.first.moduleName, config.first.sendName,
                  config.first.recvName);
      }
    }
  }
  return true;
}

void PipelineModule::go() {
  while (true) {
    std::string uri =
        "/home/wangxt/workspace/projects/flowengine/tests/data/output.json";
    startPipeline(uri);
    std::this_thread::sleep_for(std::chrono::minutes(15));
    // break;
  }
}

} // namespace module
