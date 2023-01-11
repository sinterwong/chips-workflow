
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
#include <unordered_map>
#include <utility>

namespace module {

StreamModule::StreamModule(Backend *ptr, const std::string &initName,
                           const std::string &initType,
                           const common::CameraConfig &_params)
    : Module(ptr, initName, initType) {
  cameraResult = CameraResult{static_cast<int>(_params.widthPixel),
                              static_cast<int>(_params.heightPixel),
                              25,
                              _params.cameraId,
                              _params.videoCode,
                              _params.flowType,
                              _params.cameraIp};
  vm = std::make_unique<VideoManager>(_params.cameraIp, _params.cameraId);
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
  message.clear();
  hash.clear();
  loop = false;

  beforeGetMessage();
  beforeForward();
  if (!vm->isRunning()) {
    return;
  }
  // FLOWENGINE_LOGGER_CRITICAL("Stream forward!");
  forward(message);
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
  return reinterpret_cast<void **>(&mat->data);
}

FrameBuf makeFrameBuf(cv::Mat &&frame, int height, int width) {
  FrameBuf temp;
  temp.write({std::make_any<std::shared_ptr<cv::Mat>>(
                 std::make_shared<cv::Mat>(std::forward<cv::Mat>(frame)))},
             {std::make_pair("void**", getPtrBuffer),
              std::make_pair("Mat", getMatBuffer)},
             &StreamModule::delBuffer,
             std::make_tuple(width, height, 3, UINT8));
  return temp;
}

void StreamModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      FLOWENGINE_LOGGER_INFO("{} StreamModule module was done!", name);
      // std::cout << name << "{} StreamModule module was done!" << std::endl;
      stopFlag.store(true);
      return;
    }
  }

  queueMessage sendMessage;
  cv::Mat frame = vm->getcvImage();
  if (frame.empty()) {
    return;
  }
  FrameBuf fbm =
      makeFrameBuf(std::move(frame), vm->getHeight(), vm->getWidth());
  int returnKey = backendPtr->pool->write(fbm);

  sendMessage.frameType = vm->getType();
  sendMessage.key = returnKey;
  sendMessage.cameraResult = cameraResult;
  sendMessage.status = 0;
  autoSend(sendMessage);
}
FlowEngineModuleRegister(StreamModule, Backend *, std::string const &,
                         std::string const &, common::CameraConfig const &);
} // namespace module
