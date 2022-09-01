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
#include "jetson/assdDet.hpp"
#include "jetson/yoloDet.hpp"

namespace module {

DetectModule::DetectModule(Backend *ptr, const std::string &initName,
                           const std::string &initType,
                           const common::AlgorithmConfig &_params,
                           const std::vector<std::string> &recv,
                           const std::vector<std::string> &send)
    : Module(ptr, initName, initType, recv, send),
      params(std::move(_params)) {

  if (params.algorithmSerial == "yolo") {
    instance = std::make_shared<infer::trt::YoloDet>(params);
  } else if (params.algorithmSerial == "assd") {
    instance = std::make_shared<infer::trt::AssdDet>(params);
  } else {
    FLOWENGINE_LOGGER_ERROR("Error algorithm serial {}", params.algorithmSerial);
  }
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
      // FLOWENGINE_LOGGER_INFO("{} DetectModule module was done!", name);
      std::cout << name << "{} Detection module was done!" << std::endl;
      stopFlag.store(true);
      return;
    }
    auto frameBufMessage = backendPtr->pool->read(buf.key);
    std::shared_ptr<cv::Mat> image =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
    if (type == "logic") {
      if (count++ < 5) {
        return;
      }
      count = 0;
      cv::Rect region{buf.logicInfo.region[0], buf.logicInfo.region[1],
                      buf.logicInfo.region[2] - buf.logicInfo.region[0],
                      buf.logicInfo.region[3] - buf.logicInfo.region[1]};
      std::shared_ptr<cv::Mat> inferImage;
      if (region.area() != 0) {
        inferImage = std::make_shared<cv::Mat>((*image)(region).clone());
      } else {
        inferImage = image;
      }
      if (params.algorithmSerial == "assd") {
        cv::cvtColor(*inferImage, *inferImage, cv::COLOR_RGB2BGR);
      }
      infer::Result ret;
      ret.shape = {inferImage->cols, inferImage->rows, 3};
      if (!instance->infer(inferImage->data, ret)) {
        continue;
      }

      for (auto &rbbox : ret.detResults) {
        // std::pair<std::string, std::array<float, 6>> b{}
        std::pair<std::string, std::array<float, 6>> b = {
            name,
            {rbbox.bbox[0] + region.x, rbbox.bbox[1] + region.y,
             rbbox.bbox[2] + region.x, rbbox.bbox[3] + region.y,
             rbbox.class_confidence, rbbox.class_id}};
        buf.algorithmResult.bboxes.emplace_back(std::move(b));
      }
    } else if (type == "algorithm") {
      for (int i = 0; i < buf.algorithmResult.bboxes.size(); i++) {
        auto &bbox = buf.algorithmResult.bboxes.at(i);
        if (bbox.first == send) {
          cv::Rect rect{static_cast<int>(bbox.second[0]),
                        static_cast<int>(bbox.second[1]),
                        static_cast<int>(bbox.second[2] - bbox.second[0]),
                        static_cast<int>(bbox.second[3] - bbox.second[1])};
          cv::Mat croppedImage = (*image)(rect).clone();
          infer::Result ret;
          ret.shape = {croppedImage.cols, croppedImage.rows, 3};
          if (!instance->infer(croppedImage.data, ret)) {
            continue;
          }
          for (auto &rbbox : ret.detResults) {
            std::pair<std::string, std::array<float, 6>> b = {
                name,
                {rbbox.bbox[0] + bbox.second[0], rbbox.bbox[1] + bbox.second[1],
                 rbbox.bbox[2] + bbox.second[0], rbbox.bbox[3] + bbox.second[1],
                 rbbox.class_confidence, rbbox.class_id}};
            buf.algorithmResult.bboxes.emplace_back(std::move(b));
          }
        }
      }
    }
    autoSend(buf);
  }
}
FlowEngineModuleRegister(DetectModule, Backend *, std::string const &,
                         std::string const &, common::AlgorithmConfig const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &);
} // namespace module