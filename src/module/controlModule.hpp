//
// Created by Wallel on 2022/2/22.
//

#ifndef __METAENGINE_STATUE_CONTROL_H
#define __METAENGINE_STATUE_CONTROL_H

#include "backend.h"
#include "boostMessage.h"
#include "common/common.hpp"
#include "module.hpp"
#include "utils/configParser.hpp"
#include "jetson/jetsonSourceModule.h"
#include "frameDifferenceModule.h"
#include "sendOutputModule.h"
#include "jetson/detectModule.h"
#include "logger/logger.hpp"
#include "thread_pool.h"
// #include "thread_pool.hpp"
// #include "BS_thread_pool.hpp"

#include <algorithm>
#include <memory>
#include <thread>

#include <any>
#include <memory>
#include <opencv2/opencv.hpp>

#include <vector>

using common::WORKER_TYPES;
namespace module {

class ControlModule{
private:
  std::string name = "Control";
  std::string type = "ControlMessage";
  std::string configPath;
  std::string configContent;
  std::vector<common::FlowConfigure> configs;
  utils::ConfigParser configParser;
  std::string uri;
  std::string sendUrl;
  std::shared_ptr<JetsonSourceModule> cap;
  BoostMessage bus;
  Backend backend{&bus};
  std::unique_ptr<thread_pool> pool;
  // std::unique_ptr<ThreadPool> pool;

  std::unordered_map<std::string, WORKER_TYPES> mapping{
    std::make_pair("smoke", WORKER_TYPES::SMOKE),
    std::make_pair("phone", WORKER_TYPES::PHONE),
    std::make_pair("fd", WORKER_TYPES::FD),
};

private:
  void startup(common::FlowConfigure& config);

  bool startStream();

public:
  // 后续的消息类型（initType)可以改成枚举类型
  ControlModule(std::string const &configPath_, std::string const &sendUrl_, size_t workers_n);

  ~ControlModule() {
  }

  bool initialize(); // 第一次读取配置文件并且启动流资源

  void go();  // 负责定时监控和一直运行下去
};
} // namespace module
#endif // __METAENGINE_STATUE_CONTROL_H
