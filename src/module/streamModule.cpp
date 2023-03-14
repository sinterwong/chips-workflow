
/**
 * @file streamModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-12-06
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "streamModule.h"
#include "logger/logger.hpp"
#include "messageBus.h"
#include <array>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace module {

StreamModule::StreamModule(backend_ptr ptr, std::string const &_name,
                           MessageType const &_type, ModuleConfig &_config)
    : Module(ptr, _name, _type) {

  config = std::unique_ptr<StreamBase>(_config.getParams<StreamBase>());
  try {
    vm = std::make_unique<VideoManager>(config->uri);
  } catch (const std::runtime_error &e) {
    FLOWENGINE_LOGGER_ERROR("initRecorder exception: ", e.what());
    std::runtime_error("StreamModule ctor has failed!");
  }
}

void StreamModule::beforeForward() {
  if (!vm->isRunning()) {
    if (vm->init()) {
      FLOWENGINE_LOGGER_INFO("StreamModule video is initialized!");
      if (vm->run()) {
        FLOWENGINE_LOGGER_INFO("StreamModule video is opened!");
      }
    } else {
      FLOWENGINE_LOGGER_ERROR(
          "StreamModule forward is failed, please check stream status!");
    }
  }
};

void StreamModule::step() {
  beforeGetMessage();
  beforeForward();
  if (!vm->isRunning()) {
    return;
  }

  // 按目前的设计只需要监测Close消息类型。
  MessageBus::returnFlag flag;
  std::string sender;
  MessageType stype = MessageType::None;
  queueMessage message;
  ptr->message->recv(name, flag, sender, stype, message, false);
  if (stype == MessageType::Close) {
    FLOWENGINE_LOGGER_INFO("{} StreamModule module was done!", name);
    stopFlag.store(true);
    return;
  }
  std::vector<forwardMessage> messages;
  forward(messages);
  afterForward();
}

void StreamModule::delBuffer(std::vector<std::any> &list) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(std::shared_ptr<cv::Mat>));
  list.clear();
}

std::any getMatBuffer(std::vector<std::any> &list, FrameBuf *buf) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(std::shared_ptr<cv::Mat>));
  auto mat = std::any_cast<std::shared_ptr<cv::Mat>>(list[0]);
  return mat;
}

std::any getPtrBuffer(std::vector<std::any> &list, FrameBuf *buf) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(std::shared_ptr<cv::Mat>));
  auto mat = std::any_cast<std::shared_ptr<cv::Mat>>(list[0]);
  return reinterpret_cast<void *>(mat->data);
}

FrameBuf makeFrameBuf(std::shared_ptr<cv::Mat> frame, int height, int width) {
  FrameBuf temp;
  temp.write({frame},
             {std::make_pair("void*", getPtrBuffer),
              std::make_pair("Mat", getMatBuffer)},
             &StreamModule::delBuffer,
             std::make_tuple(width, height, 3, UINT8));
  return temp;
}

void StreamModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} StreamModule module was done!", name);
      stopFlag.store(true);
      return;
    }
  }

  queueMessage sendMessage;
  auto frame = vm->getcvImage();
  if (!frame || frame->empty()) {
    FLOWENGINE_LOGGER_WARN("StreamModule get frame is failed!");
    return;
  }
  FrameBuf fbm = makeFrameBuf(frame, vm->getHeight(), vm->getWidth());
  int returnKey = ptr->pool->write(fbm);

  // 报警时所需的视频流的信息
  AlarmInfo alarmInfo;
  alarmInfo.cameraId = config->cameraId;
  alarmInfo.cameraIp = config->uri;
  alarmInfo.height = vm->getHeight();
  alarmInfo.width = vm->getWidth();

  sendMessage.frameType = vm->getType();
  sendMessage.key = returnKey;
  // sendMessage.cameraResult = cameraResult;
  sendMessage.alarmInfo = std::move(alarmInfo);
  sendMessage.status = 0;
  autoSend(sendMessage);
}
FlowEngineModuleRegister(StreamModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
