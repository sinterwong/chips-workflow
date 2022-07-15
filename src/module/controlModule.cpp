//
// Created by Wallel on 2022/2/22.
//

#include "controlModule.hpp"
#include "logger/logger.hpp"
#include <chrono>
#include <memory>

namespace module {

ControlModule::ControlModule(std::string const &configPath_,
                             std::string const &sendUrl_, size_t workers_n)
    : configPath(configPath_), sendUrl(sendUrl_) {
  pool = std::unique_ptr<thread_pool>(new thread_pool());
  pool->start(workers_n);
  // pool = std::unique_ptr<ThreadPool>(new ThreadPool (workers_n));
}

void ControlModule::startup(common::FlowConfigure &config) {
  common::AlarmInfo alarmInfo;
  alarmInfo.alarmDetails = config.alarmType;
  alarmInfo.alarmType = config.alarmType;
  alarmInfo.cameraId = config.cameraId;
  alarmInfo.provinceId = config.provinceId;
  alarmInfo.cityId = config.cityId;
  alarmInfo.regionId = config.regionId;
  alarmInfo.location = config.location;
  alarmInfo.height = config.height;
  alarmInfo.width = config.width;
  alarmInfo.cameraIp = config.cameraIp;
  alarmInfo.hostId = config.hostId;
  alarmInfo.stationId = config.stationId;
  std::string sendName = alarmInfo.alarmType + "Send";
  FLOWENGINE_LOGGER_INFO("Startup {} module", alarmInfo.alarmType);
  switch (mapping[alarmInfo.alarmType]) {
  case WORKER_TYPES::SMOKE: { // Smoking
    std::shared_ptr<FrameDifferenceModule> smoke(
        new FrameDifferenceModule(&backend, alarmInfo.alarmType, "FrameMessage",
                                  {"Camera", name}, {sendName}));
    pool->submit(&FrameDifferenceModule::go, smoke);

    // pool->enqueue(&FrameDifferenceModule::go, smoke);
    break;
  }
  case WORKER_TYPES::PHONE: { // Calling
    std::shared_ptr<FrameDifferenceModule> phone(
        new FrameDifferenceModule(&backend, alarmInfo.alarmType, "FrameMessage",
                                  {"Camera", name}, {sendName}));
    pool->submit(&FrameDifferenceModule::go, phone);

    // pool->enqueue(&FrameDifferenceModule::go, phone);
    break;
  }
  default: {
    break;
  }
  }
  cap->addSendModule(alarmInfo.alarmType);
  std::shared_ptr<SendOutputModule> output(
      new SendOutputModule(&backend, sendUrl, alarmInfo, sendName,
                           "FrameMessage", {alarmInfo.alarmType, name}));
  pool->submit(&SendOutputModule::go, output);

  // pool->enqueue(&SendOutputModule::go, output);
  FLOWENGINE_LOGGER_INFO("Startup {} module", sendName);
}

bool ControlModule::startStream() {
  if (configs.empty()) {
    FLOWENGINE_LOGGER_ERROR("ControlModule.initialize: config file cannot be "
                            "empty during initialization!");
    return false;
  }
  cap = std::shared_ptr<JetsonSourceModule>(
      new JetsonSourceModule{&backend,
                             configs[0].cameraIp,
                             configs[0].width,
                             configs[0].height,
                             configs[0].videoCode,
                             "Camera",
                             "FrameMessage",
                             {},
                             {}});

  pool->submit(&JetsonSourceModule::go, cap);
  return true;
}

bool ControlModule::initialize() {
  // 读取配置文件
  if (!configParser.readFile(configPath, configContent)) {
    FLOWENGINE_LOGGER_ERROR(
        "ControlModule.initialize: read config file is failed!");
    return false;
  }

  if (!configParser.parseConfig(configContent.c_str(), configs)) {
    FLOWENGINE_LOGGER_ERROR(
        "ControlModule.initialize: parse config file is failed!");
    return false;
  };
  if (configs.empty()) {
    FLOWENGINE_LOGGER_ERROR("ControlModule.initialize: config file cannot be "
                            "empty during initialization!");
    return false;
  }

  // run streaming
  startStream();

  // pool->enqueue(&JetsonSourceModule::go, cap);
  for (int i = 0; i < configs.size(); i++) {
    // 启动算法
    startup(configs[i]);
  }
  configs.clear(); // 及时清空configs
  return true;
}

void ControlModule::go() {
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(8));
    // 读取配置文件
    if (!configParser.readFile(configPath, configContent)) {
      continue;
    }
    if (!configParser.parseConfig(configContent.c_str(), configs)) {
      continue;
    };
    FLOWENGINE_LOGGER_INFO("Configure was updated!");
    if (configs.empty()) {
      continue;
    }
    if (!cap) {
      if (!startStream()) {
        FLOWENGINE_LOGGER_WARN("Start streaming was failed!");
      }
    }
    for (int i = 0; i < configs.size(); i++) {
      if (!configs[i].status) {
        FLOWENGINE_LOGGER_ERROR("Close {} module and {}Send module....",
                                configs[i].alarmType, configs[i].alarmType);
        cap->delSendModule(configs[i].alarmType);
        backend.message->send(name, configs[i].alarmType, type, {});
        backend.message->send(name, configs[i].alarmType + "Send", type, {});
      } else {
        startup(configs[i]);
      }
    }
    configs.clear(); // 及时清空configs

    // backend.message->send(name, "smoke", type, {});
    // backend.message->send(name, "smokeSend", type, {});
    // backend.message->send(name, "phone", type, {});
    // backend.message->send(name, "phoneSend", type, {});
    // break;
  }
}

} // namespace module
