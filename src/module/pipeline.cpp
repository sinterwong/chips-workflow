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
#include "factory.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

#include "module_utils.hpp"

namespace module {

PipelineModule::PipelineModule(std::string const &config_path_,
                               size_t workers_n)
    : config_path(config_path_) {
  pool = std::unique_ptr<thread_pool>{std::make_unique<thread_pool>()};
  pool->start(workers_n);
}

bool PipelineModule::submitModule(ModuleInfo const &info,
                                  ModuleConfig const &config) {
  atm[info.moduleName] = ObjectFactory::createObject<Module>(
      info.className, backendPtr, info.moduleName, info.moduleType, config);

  if (atm[info.moduleName] == nullptr) {
    FLOWENGINE_LOGGER_ERROR("Module {} fails to be started!", info.moduleName);
    return false;
  }
  atm[info.moduleName]->addRecvModule(name); // 关联管理员模块
  pool->submit(&Module::go, atm.at(info.moduleName));
  return true;
}

bool PipelineModule::parseConfigs(std::string const &path,
                                  std::vector<PipelineParams> &pipelines) {
  if (!configParser.parseConfig(path, pipelines)) {
    FLOWENGINE_LOGGER_INFO("config parse: parseParams is failed!");
    return false;
  }

  // 写入
  if (!utils::writeJson("{}", path)) {
    FLOWENGINE_LOGGER_INFO("config parse: clean json file is failed!");
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
  backendPtr->message->send(name, moduleName, MessageType::Close,
                            queueMessage());

  // 从atm中删除对象
  atm.erase(moduleName);
}

bool PipelineModule::startPipeline() {

  // 清除已经停止的模块（针对摄像头等外部因素）
  atm.erase(
      std::remove_if(atm.begin(), atm.end(),
                     [](const auto &kv) { return kv.second->stopFlag.load(); }),
      atm.end());

  std::vector<PipelineParams> pipelines;
  if (!parseConfigs(config_path, pipelines)) {
    FLOWENGINE_LOGGER_ERROR("parse config error");
    return false;
  }
  std::vector<std::string> currentModules;
  std::vector<ModuleInfo> moduleRelations;
  // run 起来所有模块并且制作所有需要关联的模块, 后续可能会有所扩展
  for (auto &pipeline : pipelines) {
    for (auto &config : pipeline) {
      currentModules.push_back(config.first.moduleName);
      moduleRelations.push_back(config.first);
      if (atm.find(config.first.moduleName) == atm.end()) {
        submitModule(config.first, config.second);
      }
    }
  }
  // 清掉已经停用的模块
  if (!currentModules.empty()) {
    std::unordered_map<std::string, module_ptr>::iterator iter;
    for (iter = atm.begin(); iter != atm.end();) {
      auto it =
          std::find(currentModules.begin(), currentModules.end(), iter->first);
      if (it == currentModules.end()) {
        // 说明该模块需要关闭（函数中存在删除atm的部分）
        // 解除关联
        detachModule(iter->first);
        // 发送终止正在 go 的消息
        // iter->second->stopFlag.store(true);
        backendPtr->message->send(name, iter->first, MessageType::Close,
                                  queueMessage());
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

void PipelineModule::run() {
  while (true) {
    // 现在的方式是一个大json，里面包含了所有的pipelines，需要自行判断模块增删改
    startPipeline();

    // 后续可以提供一系列的路由，如新增模块，删除模块，单帧推理等
    std::this_thread::sleep_for(std::chrono::seconds(30));
    // terminate();  // 终止所有任务
    // break;
  }
}

} // namespace module
