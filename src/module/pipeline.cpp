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
#include "common/factory.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
  if (!atm[info.moduleName]) {
    FLOWENGINE_LOGGER_ERROR("{} is failed to start!", info.moduleName);
    return false;
  }
  atm[info.moduleName]->addRecvModule(name); // 关联管理员模块
  pool->submit(&Module::go, atm.at(info.moduleName));
  return true;
}

bool PipelineModule::submitAlgo(std::string const &name,
                                AlgoConfig const &config) {
  backendPtr->algo->registered(name, config);
  return true;
}

bool PipelineModule::stopAlgo(std::string const &name) {
  backendPtr->algo->unregistered(name);
  return true;
}

bool PipelineModule::parseConfigs(std::string const &path,
                                  std::vector<PipelineParams> &pipelines,
                                  std::vector<AlgorithmParams> &algorithms) {
  if (!configParser.parseConfig(path, pipelines, algorithms)) {
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
  // 关掉与之关联的消息接收与消息发送
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

  // 从atm中删除对象（这里实际的对象不会立即销毁，因为在线程池中还有一个引用，因此会等到线程终止后才会真正销毁）
  atm.erase(moduleName);
}

void PipelineModule::removeInactiveModules() {
  std::unordered_map<std::string, module_ptr>::iterator it;
  for (it = atm.begin(); it != atm.end();) {
    if (!it->second->isRunning()) {
      detachModule(it->first);
      backendPtr->message->send(name, it->first, MessageType::Close,
                                queueMessage());
      it = atm.erase(it);
    } else {
      ++it;
    }
  }
}

bool PipelineModule::loadConfigs(std::vector<PipelineParams> &pipelines,
                                 std::vector<AlgorithmParams> &algorithms) {

  if (!parseConfigs(config_path, pipelines, algorithms)) {
    FLOWENGINE_LOGGER_ERROR("parse config error");
    return false;
  }
  return true;
}

void PipelineModule::updateAlgorithms(
    const std::vector<AlgorithmParams> &algorithms) {
  std::vector<std::string> activedAlgos, recentAlgos, willStopAlgos;
  // 获取已经在活动中的算法列表
  backendPtr->algo->getActiveAlgorithms(activedAlgos);

  // 获取最新要启动的算法列表
  recentAlgos.reserve(algorithms.size()); // 预留空间，优化性能
  std::transform(algorithms.begin(), algorithms.end(),
                 std::back_inserter(recentAlgos),
                 [](const auto &pair) { return pair.first; });

  // 比较出需要关闭的算法列表
  std::copy_if(activedAlgos.begin(), activedAlgos.end(),
               std::back_inserter(willStopAlgos),
               [&recentAlgos](const std::string &algo) {
                 return std::find(recentAlgos.begin(), recentAlgos.end(),
                                  algo) == recentAlgos.end();
               });

  // 启动算法
  for (auto const &algo : algorithms) {
    submitAlgo(algo.first, algo.second);
  }

  // 关停目前没有使用的算法
  for (auto const &name : willStopAlgos) {
    stopAlgo(name);
  }
}

void PipelineModule::updatePipelines(
    const std::vector<PipelineParams> &pipelines) {
  std::vector<std::string> currentModules;
  std::vector<ModuleInfo> moduleRelations;
  for (auto &pipeline : pipelines) {
    for (auto &config : pipeline) {
      currentModules.push_back(config.first.moduleName);
      moduleRelations.push_back(config.first);

      if (atm.find(config.first.moduleName) == atm.end()) {
        submitModule(config.first, config.second);
      }
    }
  }

  // 模块已经全部启动，将模块关联
  for (auto &mc : moduleRelations) {
    attachModule(mc.moduleName, mc.sendName, mc.recvName);
  }

  // 关闭停用的组件
  if (!currentModules.empty()) {
    std::unordered_map<std::string, module_ptr>::iterator iter;
    for (iter = atm.begin(); iter != atm.end();) {
      auto it =
          std::find(currentModules.begin(), currentModules.end(), iter->first);
      if (it == currentModules.end()) {
        // 解除关联
        detachModule(iter->first);
        // 发送终止正在 go 的消息
        backendPtr->message->send(name, iter->first, MessageType::Close,
                                  queueMessage());
        // 从atm中删除对象
        iter = atm.erase(iter);
      } else {
        ++iter;
      }
    }
  }
}

/**
 * @brief 根据配置文件的更新启动或者关闭pipeline
 * 1. 解析配置文件，获取模块关联信息和各个模块的配置参数
 * 2. 将配置中没有启动的模块和算法启动
 * 3. 建立模块之间的关系
 * 4. 关闭已经停止的模块以及前端已经关闭的模块
 * 5. 启动算法模块
 * 6. 关闭停用的算法 TODO
 *
 * @return true
 * @return false
 */
bool PipelineModule::startPipeline() {
  // 更新模块之前，首先删除已经停用的模块
  removeInactiveModules();

  // 读取最新配置
  std::vector<PipelineParams> pipelines;
  std::vector<AlgorithmParams> algorithms;
  if (!loadConfigs(pipelines, algorithms)) {
    return false;
  }
  if (pipelines.empty()) {
    return true; // 没有最新的配置
  }

  // 更新算法
  updateAlgorithms(algorithms);

  // 更新最新的pipelines
  updatePipelines(pipelines);
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
