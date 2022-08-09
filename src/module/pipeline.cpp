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

namespace module {
using common::ModuleType;

PipelineModule::PipelineModule(std::string const &config_, std::string const &sendUrl_, size_t workers_n)
    : config(config_), sendUrl(sendUrl_) {
  pool = std::unique_ptr<thread_pool>(new thread_pool());
  pool->start(workers_n);
  // pool = std::unique_ptr<BS::thread_pool>(new BS::thread_pool(workers_n));

  // 启动SendOutput模块，此模块一个进程只此一个
  ModuleConfigure config{ModuleType::Output, "output", "", ""};
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
  case ModuleType::Stream: { // Stream
    atm[config.moduleName] = std::shared_ptr<JetsonSourceModule>(
        new JetsonSourceModule(&backend, config.moduleName, "FrameMessage",
                               paramsConfig.GetCameraConfig(), {name}, {}));
    pool->submit(&JetsonSourceModule::go, atm.at(config.moduleName));
    break;
  }
  case ModuleType::Output: { // Output
    atm[config.moduleName] = std::shared_ptr<SendOutputModule>(
        new SendOutputModule(&backend, config.moduleName, "OutputMessage",
                             paramsConfig.GetOutputConfig(), {name}, {}));
    pool->submit(&SendOutputModule::go, atm.at(config.moduleName));
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
    atm.at(moduleName)->addSendModule(sendModule);
  }

  if (!recvModule.empty()) {
    atm.at(moduleName)->addRecvModule(recvModule);
  }
}

void PipelineModule::delModule(std::string const &moduleName) {

  auto iter = atm.find(moduleName);
  if (iter == atm.end()) {
    FLOWENGINE_LOGGER_ERROR("{} is not runing", moduleName);
    return ;
  }
  // 删除模块之前需要先解除所有关联
  for (auto &sm : atm.at(moduleName)->getSendModule()) {
    atm.at(sm)->delRecvModule(moduleName);
  };

  for (auto &rm : atm.at(moduleName)->getRecvModule()) {
    if (rm == name) {  // 保留pipeline控制模块
      continue;
    }
    atm.at(rm)->delSendModule(moduleName);
  };
  // 发送终止消息
  backend.message->send(name, moduleName, type, queueMessage());
}

bool PipelineModule::startPipeline() {
  std::vector<
      std::vector<std::pair<ModuleConfigure, ParamsConfig>>> pipelines;
  if (!parseConfigs(config, pipelines)) {
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
    startPipeline();
    std::this_thread::sleep_for(std::chrono::seconds(50));
    // terminate();  // 终止所有任务
    // break;
  }
}

} // namespace module
