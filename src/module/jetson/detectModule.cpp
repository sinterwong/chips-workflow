//
// Created by Wallel on 2022/3/1.
//

#include "jetson/detectModule.h"
#include "backend.h"
#include "inference.h"
#include <cassert>
#include <opencv2/imgcodecs.hpp>

namespace module {
cv::Rect get_rect(cv::Mat &img, int width, int height,
                  std::array<float, 4> bbox) {
  int l, r, t, b;
  float r_w = width / (img.cols * 1.0);
  float r_h = height / (img.rows * 1.0);
  if (r_h > r_w) {
    l = bbox[0] - bbox[2] / 2.f;
    r = bbox[0] + bbox[2] / 2.f;
    t = bbox[1] - bbox[3] / 2.f - (height - r_w * img.rows) / 2;
    b = bbox[1] + bbox[3] / 2.f - (height - r_w * img.rows) / 2;
    l = l / r_w;
    r = r / r_w;
    t = t / r_w;
    b = b / r_w;
  } else {
    l = bbox[0] - bbox[2] / 2.f - (width - r_h * img.cols) / 2;
    r = bbox[0] + bbox[2] / 2.f - (width - r_h * img.cols) / 2;
    t = bbox[1] - bbox[3] / 2.f;
    b = bbox[1] + bbox[3] / 2.f;
    l = l / r_h;
    r = r / r_h;
    t = t / r_h;
    b = b / r_h;
  }
  return cv::Rect(l, t, r - l, b - t);
}

DetectModule::DetectModule(Backend *ptr, const std::string &initName,
                           const std::string &initType,
                           const common::ParamsConfig _params,
                           const std::vector<std::string> &recv,
                           const std::vector<std::string> &send,
                           const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool), params(_params) {
      
  inferParams.batchSize = 1;
  inferParams.numAnchors = 25200;
  inferParams.numClasses = 80;
  inferParams.inputTensorNames.push_back("images");
  inferParams.outputTensorNames.push_back("output");
  inferParams.inputShape = {640, 640, 3};
  inferParams.serializedFilePath = params.modelDir + "/yolov5s.engine";
  inferParams.originShape = params.originShape;
  inferParams.scaling =
      float(inferParams.inputShape[0]) /
      (float)std::max(inferParams.originShape[0], inferParams.originShape[1]);

  instance = std::make_shared<infer::trt::DetctionInfer>(inferParams);
  instance->initialize();
}

DetectModule::~DetectModule() {}

void DetectModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  if (!instance) {
    std::cout << "instance is not init!!!" << std::endl;
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      std::cout << "get Control!" << std::endl;
    } else if (type == "FrameMessage") {
      auto frameBufMessage = backendPtr->pool.read(buf.key);
      void *data = std::any_cast<void *>(frameBufMessage.read("void*"));
      auto frame =
          std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
      // TODO
      // 想办法返回结果信息，或者画到Mat上也可以，然后直接send出去就可以呈现出来了
      infer::Result ret;
      instance->infer(data, ret);

      for (size_t j = 0; j < ret.detResults[0].size(); j++) {
        cv::Rect r = get_rect(*frame, inferParams.inputShape[0],
                              inferParams.inputShape[1], ret.detResults[0][j].bbox);
        cv::rectangle(*frame, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
        cv::putText(*frame, std::to_string((int)ret.detResults[0][j].class_id),
                    cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2,
                    cv::Scalar(0xFF, 0xFF, 0xFF), 2);
      }
      autoSend(buf);
      // count++;
    }
  }
  // if (count > 300) {
  //   // check if the user quit
  //   stopFlag.store(true);
  // }
}
} // namespace module