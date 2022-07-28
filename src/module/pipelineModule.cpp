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
#include "jetson/classifierModule.h"
#include "jetson/jetsonSourceModule.h"
#include "sendOutputModule.h"
#include <chrono>
#include <memory>

namespace module {

PipelineModule::PipelineModule(std::string const &sendUrl_, size_t workers_n)
    : sendUrl(sendUrl_) {
  pool = std::unique_ptr<thread_pool>(new thread_pool());
  pool->start(workers_n);

  // 启动SendOutput模块，此模块一个进程只此一个
  common::ModuleConfigure config{"SendOutput", "", ""};
  common::SendConfig sendConfig = {sendUrl};
  common::ParamsConfig paramsConfig{sendConfig};
  submitModule(config, paramsConfig);
}

bool PipelineModule::submitModule(common::ModuleConfigure const &config,
                                  common::ParamsConfig const &paramsConfig) {
  switch (paramsConfig.GetKind()) {
  case common::ModuleType::Detection: { // Detection
    atm[config.moduleName] = std::shared_ptr<DetectModule>(
        new DetectModule(&backend, config.moduleName, "AlgorithmMessage",
                         paramsConfig.GetAlgorithmConfig(), {name}, {}));
    pool->submit(&DetectModule::go, atm[config.moduleName]);
    break;
  }
  case common::ModuleType::Classifier: { // Classifier
    atm[config.moduleName] = std::shared_ptr<ClassifierModule>(
        new ClassifierModule(&backend, config.moduleName, "AlgorithmMessage",
                             paramsConfig.GetAlgorithmConfig(), {name}, {}));
    pool->submit(&ClassifierModule::go, atm[config.moduleName]);
    break;
  }
  case common::ModuleType::Stream: { // Stream
    atm[config.moduleName] = std::shared_ptr<JetsonSourceModule>(
        new JetsonSourceModule(&backend, config.moduleName, "FrameMessage",
                               paramsConfig.GetCameraConfig(), {}, {}));
    pool->submit(&JetsonSourceModule::go, atm[config.moduleName]);
    break;
  }
  case common::ModuleType::Output: { // Output
    atm[config.moduleName] = std::shared_ptr<SendOutputModule>(
        new SendOutputModule(&backend, config.moduleName, "OutputMessage",
                             paramsConfig.GetSendConfig(), {name}, {}));
    pool->submit(&SendOutputModule::go, atm[config.moduleName]);
    break;
  }
  default: {
    break;
  }
  }
  addModule(config.moduleName, config.sendName, config.recvName);
  return true;
}

bool PipelineModule::getParamConfig(std::string const &name,
                                    common::ParamsConfig &config) {
  // 通过模块的名称获取模块的配置
  
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

bool PipelineModule::startPipeline(common::FlowConfigure const &config) {
  common::WorkerTypes type = typeMapping[config.alarmType];
  switch (type) {
  case common::WorkerTypes::Calling: { // pipeline calling
    // Step1: check gesture classifier status
    if (atm.find("web-camera-01") == atm.end()) {
      common::ModuleConfigure moduleConfigure{"web-camera-01", "HandDet", ""};
      common::ParamsConfig paramsConfig = common::CameraConfig{
          1920,
          1080,
          "web-camera-01",
          "h264",
          "rtsp",
          "/home/wangxt/workspace/projects/flowengine/tests/data/sample_1080p_h264.mp4"};
      submitModule(moduleConfigure, paramsConfig);
    } else {
      addModule("web-camera-01", "HandDet", "");
    }

    if (atm.find("HandDet") == atm.end()) {
      common::ModuleConfigure moduleConfigure{"HandDet", "PhoneCls",
                                              "web-camera-01"};
      common::ParamsConfig paramsConfig =
          common::AlgorithmConfig{common::ModuleType::Detection,
                                  "/home/wangxt/workspace/projects/flowengine/tests/data/yolov5s.engine",
                                  {"images"},
                                  {"output"},
                                  {640, 640, 3},
                                  80, 
                                  25200};
      submitModule(moduleConfigure, paramsConfig);
    } else {
      addModule("HandDet", "PhoneCls", "web-camera-01");
    }
    if (atm.find("PhoneCls") == atm.end()) {
      common::ModuleConfigure moduleConfigure{"PhoneCls", "SendOutput",
                                              "HandDet"};
      common::ParamsConfig paramsConfig = common::AlgorithmConfig{
          common::ModuleType::Classifier,
          "/home/wangxt/workspace/projects/flowengine/tests/data/"
          "RepVGG-C0_ce_dog-cat_96x96_93.834.engine",
          {"input"},
          {"output"},
          {96, 96, 3},
          2};
      submitModule(moduleConfigure, paramsConfig);
    } else {
      addModule("PhoneCls", "SendOutput", "HandDet");
    }
    break;
  }
  case common::WorkerTypes::CatDog: { // pipeling cat dog
    break;
  }
  case common::WorkerTypes::Smoking: { // pipeling smoking
    break;
  }
  default: {
    break;
  }
  }

  return true;
}

void PipelineModule::go() {
  while (true) {
    common::FlowConfigure config;
    startPipeline(config);
    std::this_thread::sleep_for(std::chrono::minutes(2));
    break;
  }
}

} // namespace module
