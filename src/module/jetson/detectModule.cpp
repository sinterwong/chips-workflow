/**
 * @file detectModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "jetson/detectModule.h"
#include "backend.h"
#include "inference.h"
#include <cassert>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <utility>

namespace module {

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
      FLOWENGINE_LOGGER_INFO("{} DetectModule module was done!", name);
      stopFlag.store(true);
    } else if (type == "FrameMessage") {
      auto frameBufMessage = backendPtr->pool.read(buf.key);
      // void *data = std::any_cast<void *>(frameBufMessage.read("void*"));
      std::shared_ptr<cv::Mat> image =
          std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

      cv::Mat croppedImage;
      cv::Rect rect;
      if (!params.region.empty()) {
        rect = cv::Rect{params.region[0], params.region[1], params.region[2],
                        params.region[3]};
        croppedImage = cv::Mat(*image, rect);
        infer::Result ret;
        ret.shape = {croppedImage.cols, croppedImage.rows, 3};
        instance->infer(croppedImage.data, ret);
        buf.width = croppedImage.cols;
        buf.height = croppedImage.rows;
        // 直接替换掉原本检测的类别属性
        std::pair<std::string, std::array<float, 6>> b = {
            name,
            {static_cast<float>(params.region[0]),
             static_cast<float>(params.region[1]),
             static_cast<float>(params.region[2]),
             static_cast<float>(params.region[3]), ret.classResult.second,
             static_cast<float>(ret.classResult.first)}};
        buf.results.bboxes.emplace_back(std::move(b));
        autoSend(buf);
      } else {
        for (auto &bbox : buf.results.bboxes) {
          if (bbox.first == send) {
            rect = cv::Rect{static_cast<int>(bbox.second[0]),
                            static_cast<int>(bbox.second[1]),
                            static_cast<int>(bbox.second[2] - bbox.second[0]),
                            static_cast<int>(bbox.second[3] - bbox.second[1])};
            croppedImage = cv::Mat(*image, rect);
            infer::Result ret;
            ret.shape = {croppedImage.cols, croppedImage.rows, 3};
            instance->infer(croppedImage.data, ret);
            buf.width = croppedImage.cols;
            buf.height = croppedImage.rows;
            for (auto &rbbox : ret.detResults) {
              // std::pair<std::string, std::array<float, 6>> b{}
              std::pair<std::string, std::array<float, 6>> b = {
                  name,
                  {rbbox.bbox[0] + bbox.second[0],
                   rbbox.bbox[1] + bbox.second[1],
                   rbbox.bbox[2] + bbox.second[0],
                   rbbox.bbox[3] + bbox.second[1], rbbox.class_confidence,
                   rbbox.class_id}};
              buf.results.bboxes.emplace_back(std::move(b));
            }
            autoSend(buf);
          }
        }
      }
    }
  }
}
} // namespace module