/**
 * @file controlModule.hpp
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

#include "backend.h"
#include "boostMessage.h"
#include "common/common.hpp"
#include "frameDifferenceModule.h"
#include "jetson/detectModule.h"
#include "jetson/jetsonSourceModule.h"
#include "logger/logger.hpp"
#include "module.hpp"
#include "sendOutputModule.h"
#include "thread_pool.h"
#include "utils/configParser.hpp"
// #include "thread_pool.hpp"
// #include "BS_thread_pool.hpp"

#include <algorithm>
#include <memory>
#include <thread>

#include <any>
#include <memory>
#include <opencv2/opencv.hpp>

#include <unordered_map>
#include <vector>

namespace module {

class PipelineModule {
private:
  std::string name = "Control";
  std::string type = "ControlMessage";
  utils::ConfigParser configParser;
  std::string sendUrl;
  Backend backend{std::unique_ptr<MessageBus>{new BoostMessage()},
                  std::unique_ptr<RouteFramePool>{new RouteFramePool(16)}};
  std::unique_ptr<thread_pool> pool;
  std::unordered_map<std::string, std::shared_ptr<Module>> atm;

private:
  bool submitModule(common::ModuleConfigure const &config,
                    common::ParamsConfig const &paramsConfig);

  bool startPipeline(common::FlowConfigure const &config);

  bool getParamConfig(std::string const &name, common::ParamsConfig &config);

  void addModule(std::string const &moduleName, std::string const &sendModule,
                 std::string const &recvModule);

public:
  PipelineModule(std::string const &sendUrl_, size_t workers_n);

  ~PipelineModule() {}

  bool initialize(); // 第一次读取配置文件并且启动流资源

  void go(); // 负责定时监控和一直运行下去
};
} // namespace module
#endif // __METAENGINE_STATUE_CONTROL_H
