/**
 * @file pipelineModule.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-07-10
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_STATUE_CONTROL_H
#define __METAENGINE_STATUE_CONTROL_H

#include "algorithmManager.hpp"
#include "backend.h"
#include "boostMessage.h"
#include "common/common.hpp"
#include "configParser.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include "thread_pool.h"

#include <algorithm>
#include <any>
#include <chrono>
#include <memory>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace module {

using common::ModuleConfig;
using common::ModuleInfo;
using infer::AlgorithmManager;
using utils::AlgorithmParams;
using utils::PipelineParams;

using module_ptr = std::shared_ptr<Module>;

class PipelineModule {
private:
  std::string name = "Administrator";
  std::string config_path;
  utils::ConfigParser configParser;
  backend_ptr backendPtr = std::make_shared<Backend>(
      std::make_unique<BoostMessage>(), std::make_unique<RouteFramePool>(6),
      std::make_unique<AlgorithmManager>());
  // std::unique_ptr<thread_pool> pool;
  std::unique_ptr<thread_pool> pool;
  std::unordered_map<std::string, module_ptr> atm;

private:
  /**
   * @brief 提交模块到线程池
   *
   * @param config
   * @param paramsConfig
   * @return true
   * @return false
   */
  bool submitModule(ModuleInfo const &info, ModuleConfig const &config);

  /**
   * @brief 终止并删除模块
   *
   * @param moduleName
   */
  void stopModule(std::string const &moduleName);

  /**
   * @brief 关联模块
   *
   * @param moduleName
   * @param sendModule
   * @param recvModule
   */
  void attachModule(std::string const &moduleName,
                    std::string const &sendModule,
                    std::string const &recvModule);

  /**
   * @brief 解除模块关联
   *
   * @param moduleName
   */
  void detachModule(std::string const &moduleName);

  /**
   * @brief 注册提交算法模块
   *
   * @param name
   * @param config
   * @return true
   * @return false
   */
  bool submitAlgo(std::string const &name, AlgoConfig const &config);

  /**
   * @brief 关闭算法
   *
   * @param name
   * @param config
   * @return true
   * @return false
   */
  bool stopAlgo(std::string const &name);

  /**
   * @brief 定时刷新算法pipeline
   *
   * @return true
   * @return false
   */
  bool startPipeline();

  bool parseConfigs(std::string const &uri, std::vector<PipelineParams> &,
                    std::vector<AlgorithmParams> &);

  void terminate() {
    // 向所有模块发送终止信号
    // for (auto iter = atm.begin(); iter != atm.end(); ++iter) {
    //   backendPtr->message->send(name, iter->first, MessageType::Close,
    //                             queueMessage());
    // }
    std::this_thread::sleep_for(std::chrono::seconds(5));
  }

public:
  PipelineModule(std::string const &config_, size_t workers_n);

  ~PipelineModule() { terminate(); }

  bool initialize(); // 第一次读取配置文件并且启动流资源

  void run(); // 开始运行
};
} // namespace module
#endif // __METAENGINE_STATUE_CONTROL_H
