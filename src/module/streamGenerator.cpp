/**
 * @file StreamGenerator.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.2
 * @date 2022-10-26
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "streamGenerator.h"

#include "logger/logger.hpp"
#include "messageBus.h"
#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <sys/types.h>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace module {

StreamGenerator::StreamGenerator(Backend *ptr, const std::string &initName,
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
}

void StreamGenerator::beforeForward() {
};

void StreamGenerator::step() {
  message.clear();
  hash.clear();
  loop = false;

  beforeGetMessage();
  beforeForward();

  forward(message);
  afterForward();
}

void StreamGenerator::delBuffer(std::vector<std::any> &list) {
  /*
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(VIDEO_FRAME_S *));
  list.clear();
  */
}

std::any getFrameInfo(std::vector<std::any> &list, FrameBuf *buf) {
  /**
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(VIDEO_FRAME_S *));
  return reinterpret_cast<void *>(std::any_cast<VIDEO_FRAME_S *>(list[0]));
  */
  return bool();
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

FrameBuf makeFrameBuf(cv::Mat &frame, int height, int width) {
  FrameBuf temp;
  temp.write({std::make_any<std::shared_ptr<cv::Mat>>(
                 std::make_shared<cv::Mat>(frame))},
             {std::make_pair("void**", getPtrBuffer),
              std::make_pair("Mat", getMatBuffer)},
             &StreamGenerator::delBuffer,
             std::make_tuple(width, height, 3, UINT8));
  return temp;
}

void StreamGenerator::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      // FLOWENGINE_LOGGER_INFO("{} JetsonSourceModule module was done!", name);
      std::cout << name << "{} StreamGenerator module was done!" << std::endl;
      stopFlag.store(true);
      return;
    }
  }
  /*
  decoder->getFrame(frame);
  queueMessage sendMessage;
  FrameBuf frameBufMessage =
      makeFrameBuf(frame, decoder->getHeight(), decoder->getWidth());
  int returnKey = backendPtr->pool->write(frameBufMessage);

  sendMessage.frameType = decoder->getType();
  sendMessage.key = returnKey;
  sendMessage.cameraResult = cameraResult;
  sendMessage.status = 0;
  autoSend(sendMessage);
  // FLOWENGINE_LOGGER_INFO("Send the frame message!");
  */
}
FlowEngineModuleRegister(StreamGenerator, Backend *, std::string const &,
                         std::string const &, common::CameraConfig const &);
} // namespace module
