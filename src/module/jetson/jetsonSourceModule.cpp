//
// Created by Wallel on 2022/2/22.
//

#include "jetsonSourceModule.h"
#include <mutex>
#include <opencv2/imgproc.hpp>

namespace module {

void delBuffer(std::vector<std::any> &list) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(uchar3 *));
  list.clear();
}

std::any getPtrBuffer(std::vector<std::any> &list, FrameBuf *buf) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(uchar3 *));
  return reinterpret_cast<void *>(std::any_cast<uchar3 *>(list[0]));
}

std::any getMatBuffer(std::vector<std::any> &list, FrameBuf *buf) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(uchar3 *));

  void *data = reinterpret_cast<void *>(std::any_cast<uchar3 *>(list[0]));
  std::shared_ptr<cv::Mat> mat =
      std::make_shared<cv::Mat>(buf->height, buf->width, CV_8UC3, data);
  cv::cvtColor(*mat, *mat, cv::COLOR_BGR2RGB);
  return mat;
}

std::any getUchar3Buffer(std::vector<std::any> &list, FrameBuf *buf) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(uchar3 *));

  return std::any_cast<uchar3 *>(list[0]);
}

FrameBuf makeFrameBuf(uchar3 *image, int height, int width) {
  FrameBuf temp;
  temp.write({std::make_any<uchar3 *>(image)},
             {std::make_pair("void*", getPtrBuffer),
              std::make_pair("Mat", getMatBuffer),
              std::make_pair("uchar3*", getUchar3Buffer)},
             delBuffer, std::make_tuple(height, width, 3, UINT8));
  return temp;
}

JetsonSourceModule::JetsonSourceModule(
    Backend *ptr, const std::string &uri, const int width, const int height,
    const std::string &codec, const std::string &initName,
    const std::string &initType, const std::vector<std::string> &recv,
    const std::vector<std::string> &send, const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool) {

  opt.height = height;
  opt.width = width;
  opt.codec = videoOptions::CodecFromStr(codec.c_str());
  opt.resource = uri;
  // inputStream = std::unique_ptr<videoSource>(videoSource::Create(opt));
  inputStream = videoSource::Create(opt);
  if (!inputStream) {
    LogError("jetson source:  failed to create input stream\n");
    exit(-1);
  }
}

void JetsonSourceModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  assert(inputStream);

  ret = inputStream->Capture(&frame, 1000);

  if (ret) {
    FrameBuf frameBufMessage = makeFrameBuf(frame, opt.height, opt.width);
    frameBufMessage.height = opt.height;
    frameBufMessage.width = opt.width;

    int returnKey = backendPtr->pool.write(frameBufMessage);

    queueMessage sendMessage;
    sendMessage.type = "RGB888";
    sendMessage.height = opt.height;
    sendMessage.width = opt.width;
    sendMessage.key = returnKey;
    autoSend(sendMessage);
    // count++;
  } else {
    if (!inputStream->IsStreaming()) {
      LogInfo("jetson source:  input steram was done.\n");
      // sendCM();
      stopFlag.store(true);
    }
  }
  // if (count > 2000) {
  //   stopFlag.store(true);
  // }
}

void JetsonSourceModule::delSendModule(std::string const &name) {
  std::lock_guard<std::mutex> lk(_m);
  auto iter = std::remove(sendModule.begin(), sendModule.end(), name);
  sendModule.erase(iter, sendModule.end());
}

void JetsonSourceModule::addSendModule(std::string const &name) {
  std::lock_guard<std::mutex> lk(_m);
  sendModule.push_back(name);
}


} // namespace module
