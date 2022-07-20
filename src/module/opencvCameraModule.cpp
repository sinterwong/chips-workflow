//
// Created by Wallel on 2022/2/22.
//

#include "opencvCameraModule.h"

namespace module {
void delOpencvShareMat(std::vector<std::any> &list) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(std::shared_ptr<cv::Mat>));

  list.clear();
}

std::any getPtrOpencvShareMat(std::vector<std::any> &list, FrameBuf *buf) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(std::shared_ptr<cv::Mat>));

  return std::any_cast<std::shared_ptr<cv::Mat>>(list[0])->data;
}

std::any getMatOpencvShareMat(std::vector<std::any> &list, FrameBuf *buf) {
  assert(list.size() == 1);
  assert(list[0].has_value());
  assert(list[0].type() == typeid(std::shared_ptr<cv::Mat>));

  return *std::any_cast<std::shared_ptr<cv::Mat>>(list[0]);
}

FrameBuf makeFrameBuf(std::shared_ptr<cv::Mat> mat) {
  FrameBuf temp;
  temp.write({std::make_any<std::shared_ptr<cv::Mat>>(mat)},
             {std::make_pair("void*", getPtrOpencvShareMat),
              std::make_pair("Mat", getMatOpencvShareMat)},
             delOpencvShareMat,
             std::make_tuple(mat->cols, mat->rows, mat->channels(), UINT8));
  return temp;
}

OpencvCameraModule::OpencvCameraModule(Backend *ptr, const std::string &file,
                                       const std::string &initName,
                                       const std::string &initType,
                                       const std::vector<std::string> &recv,
                                       const std::vector<std::string> &send,
                                       const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool) {
  readFile = true;
  fileName = file;
  cameraNumber = -1;

  cap = cv::VideoCapture(fileName);
}

OpencvCameraModule::OpencvCameraModule(Backend *ptr, const int capNumber,
                                       const std::string &initName,
                                       const std::string &initType,
                                       const std::vector<std::string> &recv,
                                       const std::vector<std::string> &send,
                                       const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool) {
  readFile = false;
  fileName = "";
  cameraNumber = capNumber;

  cap = cv::VideoCapture(cameraNumber);
}

void OpencvCameraModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  assert(cap.isOpened());
  frame = std::make_shared<cv::Mat>();
  ret = cap.read(*frame);
  std::printf("Get frame!\n");
  if (ret) {
    FrameBuf frameBufMessage = makeFrameBuf(frame);

    int returnKey = backendPtr->pool.write(frameBufMessage);

    queueMessage sendMessage;
    sendMessage.frameType = "BGA888";
    sendMessage.height = frame->rows;
    sendMessage.width = frame->cols;
    sendMessage.key = returnKey;
    autoSend(sendMessage);
  } else {
    std::printf("Warning, The capture is not opened!\n");
  }
}

void OpencvCameraModule::afterForward() {
  if (readFile) {
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
  }
}
} // namespace module