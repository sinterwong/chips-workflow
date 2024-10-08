
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
#include "streamModule.hpp"
#include "framePool.hpp"
#include "logger/logger.hpp"
#include "messageBus.hpp"
#include <array>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "videoDecoderPool.hpp"

namespace module {

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

void delBuffer(std::vector<std::any> &list) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(std::shared_ptr<cv::Mat>));
  list.clear();
}

frame_ptr makeFrameBuf(std::shared_ptr<cv::Mat> frame, int height, int width) {
  FrameBuf temp;
  temp.write({frame},
             {std::make_pair("void*", getPtrBuffer),
              std::make_pair("Mat", getMatBuffer)},
             &delBuffer);
  return std::make_shared<FrameBuf>(temp);
}

StreamModule::StreamModule(backend_ptr ptr, std::string const &_name,
                           MessageType const &_type, ModuleConfig &config_)
    : Module(ptr, _name, _type, *config_.getParams<StreamBase>()) {

  config = std::make_unique<StreamBase>(*config_.getParams<StreamBase>());
  if (ptr->pools->registered(name, 2)) {
    FLOWENGINE_LOGGER_INFO("{} stream pool registering is successful", name);
  } else {
    FLOWENGINE_LOGGER_CRITICAL("{} stream pool registering is failed!", name);
    throw std::runtime_error("StreamModule ctor has failed!");
  }
}

void StreamModule::messageListener() {
  // 在此处监听外界的消息，这样的话就可以放心大胆的实现轮询逻辑了
  while (!stopFlag.load()) {
    // 检查外界消息
    std::string sender;
    MessageType stype = MessageType::None;
    queueMessage message;
    ptr->message->recv(name, sender, stype, message);
    if (stype == MessageType::Close) {
      stopFlag.store(true);
      return;
    }
    // 避免内旋过分占用资源
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void StreamModule::go() {
  std::thread listenerThread(&StreamModule::messageListener, this);
  // 或者保存该线程，并在适当的时机进行 join()
  listenerThread.detach();

  while (!stopFlag.load()) {
    step();
  }
}

void StreamModule::step() {
  // 轮询逻辑
  if (!decoder) { // 解码器不在工作状态
    beforeForward();
  }

  if (!decoder || !decoder->isRunning()) {
    return;
  }
  auto currentTime = std::chrono::high_resolution_clock::now();
  if (config->runTime == 0 || currentTime < endTime) {
    startup();
  } else {
    afterForward();
  }
}

void StreamModule::beforeForward() {
  // 此处会悬停等待解码器（即使是收到了Close信号依然会继续排队）
  decoder = video::FIFOVideoDecoderPool::getInstance().acquire();

  // 启动解码器
  if (!decoder->start(config->uri)) {
    FLOWENGINE_LOGGER_ERROR("StreamModule is failed to open {}!", config->uri);
    // 归还解码器
    video::FIFOVideoDecoderPool::getInstance().release(std::move(decoder));
    decoder = nullptr;
    // 等待一会再回去排队
    std::this_thread::sleep_for(std::chrono::seconds(2));
    return;
  }
  startTime = std::chrono::high_resolution_clock::now(); // 初始化开始时间
  endTime = startTime + std::chrono::seconds(config->runTime); // 初始化结束时间
  FLOWENGINE_LOGGER_INFO("StreamModule video is opened {}!", config->uri);
};

void StreamModule::startup() {

  queueMessage sendMessage;
  auto frame = decoder->getcvImage();

  if (!frame || frame->empty()) {
    FLOWENGINE_LOGGER_WARN("{} StreamModule get frame is failed!", name);
    return;
  }

  frame_ptr fbm =
      makeFrameBuf(frame, decoder->getHeight(), decoder->getWidth());
  int returnKey = ptr->pools->write(name, fbm);

  // 报警时所需的视频流的信息
  AlarmInfo alarmInfo;
  alarmInfo.cameraId = config->cameraId;
  alarmInfo.cameraIp = config->uri;
  alarmInfo.height = decoder->getHeight();
  alarmInfo.width = decoder->getWidth();
  sendMessage.frameType = decoder->getType();
  sendMessage.steramName = name;
  sendMessage.key = returnKey;
  // sendMessage.cameraResult = cameraResult;
  sendMessage.alarmInfo = std::move(alarmInfo);
  sendMessage.status = 0;

  autoSend(sendMessage);
  // std::this_thread::yield();
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void StreamModule::afterForward() {
  // 此时意味着需要对decoder进行归还
  if (decoder) {
    video::FIFOVideoDecoderPool::getInstance().release(std::move(decoder));
    decoder = nullptr;
    FLOWENGINE_LOGGER_DEBUG("{} has finished!", name);
  }
}
} // namespace module
