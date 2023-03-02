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

OpencvCameraModule::OpencvCameraModule(backend_ptr ptr, const std::string &file,
                                       std::string const &name,
                                       MessageType const &type)
    : Module(ptr, name, type) {
  readFile = true;
  fileName = file;
  cameraNumber = -1;

  cap = cv::VideoCapture(fileName);
}

OpencvCameraModule::OpencvCameraModule(backend_ptr ptr, const int capNumber,
                                       std::string const &name,
                                       MessageType const &type)
    : Module(ptr, name, type) {
  readFile = false;
  fileName = "";
  cameraNumber = capNumber;

  cap = cv::VideoCapture(cameraNumber);
}

void OpencvCameraModule::forward(std::vector<forwardMessage> &message) {
  assert(cap.isOpened());
  frame = std::make_shared<cv::Mat>();
  ret = cap.read(*frame);
  std::printf("Get frame!\n");
  if (ret) {
    FrameBuf frameBufMessage = makeFrameBuf(frame);

    int returnKey = ptr->pool->write(frameBufMessage);

    queueMessage sendMessage;

    // 报警时所需的视频流的信息
    AlarmInfo alarmInfo;
    alarmInfo.cameraId = 0;
    alarmInfo.cameraIp = fileName;
    alarmInfo.height = frame->rows;
    alarmInfo.width = frame->cols;

    sendMessage.frameType = ColorType::BGR888;
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
FlowEngineModuleRegister(OpencvCameraModule, backend_ptr, std::string const &,
                         std::string const &, MessageType const &);
} // namespace module