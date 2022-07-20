/**
 * @file classifierModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-07-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "jetson/classifierModule.h"
#include "backend.h"
#include "inference.h"
#include "jetson/classifier.hpp"
#include <cassert>
#include <opencv2/imgcodecs.hpp>

namespace module {

ClassifierModule::ClassifierModule(Backend *ptr, const std::string &initName,
                                   const std::string &initType,
                                   const common::ParamsConfig _params,
                                   const std::vector<std::string> &recv,
                                   const std::vector<std::string> &send,
                                   const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool), params(_params) {

  // 算法配置信息，之后可以存入数据库
  inferParams.batchSize = 1;
  inferParams.numClasses = 2;
  inferParams.inputTensorNames.push_back("input");
  inferParams.outputTensorNames.push_back("output");
  inferParams.inputShape = {640, 640, 3};
  inferParams.serializedFilePath = params.modelDir + "/yolov5s.engine";

  instance = std::make_shared<infer::trt::ClassifierInfer>(inferParams);
  instance->initialize();
}

ClassifierModule::~ClassifierModule() {}

void ClassifierModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  if (!instance) {
    std::cout << "instance is not init!!!" << std::endl;
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      FLOWENGINE_LOGGER_INFO("{} ClassifierModule module was done!", name);
      stopFlag.store(true);
    } else if (type == "FrameMessage") {

      auto frameBufMessage = backendPtr->pool.read(buf.key);
      std::shared_ptr<cv::Mat> image =
          std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

      // 如果存在 region, 就基于提供的region进行分类,
      // 暂时认定为这个模块就是算法最上游

      cv::Mat croppedImage;
      cv::Rect rect;
      if (!params.region.empty()) {
        rect = cv::Rect{params.region[0], params.region[1], params.region[2],
                        params.region[3]};
        croppedImage = cv::Mat(*image, rect);
        infer::Result ret;
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
          // 如果存在上游，就只需要分类基于上游bbox给一个类别和置信度即可
          if (bbox.first == send) {
            rect = cv::Rect{static_cast<int>(bbox.second[0]),
                            static_cast<int>(bbox.second[1]),
                            static_cast<int>(bbox.second[2] - bbox.second[0]),
                            static_cast<int>(bbox.second[3] - bbox.second[1])};
            croppedImage = cv::Mat(*image, rect);
            infer::Result ret;
            instance->infer(croppedImage.data, ret);
            buf.width = croppedImage.cols;
            buf.height = croppedImage.rows;
            // 直接替换掉原本检测的类别属性
            bbox.first = name;
            bbox.second[5] = ret.classResult.first;  // confidence
            bbox.second[4] = ret.classResult.second; // class_id
            autoSend(buf);
          }
        }
      }
    }
  }
}
} // namespace module