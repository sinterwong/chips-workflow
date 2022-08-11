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
#include "logger/logger.hpp"
#include <cassert>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

namespace module {

ClassifierModule::ClassifierModule(Backend *ptr, const std::string &initName,
                                   const std::string &initType,
                                   const common::AlgorithmConfig &_params,
                                   const std::vector<std::string> &recv,
                                   const std::vector<std::string> &send,
                                   const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool), params(_params) {

  instance = std::make_shared<infer::trt::ClassifierInfer>(params);
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
      // FLOWENGINE_LOGGER_INFO("{} ClassifierModule module was done!", name);
      std::cout << name << "{} Classifier module was done!" << std::endl;
      stopFlag.store(true);
      return;
    }
    auto frameBufMessage = backendPtr->pool->read(buf.key);
    std::shared_ptr<cv::Mat> image =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
    if (type == "FrameMessage") {
      if (count++ < 5) {
        return;
      }
      count = 0;
      std::shared_ptr<cv::Mat> inferImage;
      if (region.area() != 0) {
        inferImage = std::make_shared<cv::Mat>((*image)(region).clone());
      } else {
        inferImage = image;
      }
      infer::Result ret;
      ret.shape = {inferImage->cols, inferImage->rows, 3};
      if (!instance->infer(inferImage->data, ret)) {
        continue;
      }
      std::pair<std::string, std::array<float, 6>> b = {
          name,
          {static_cast<float>(region.x), static_cast<float>(region.y),
           static_cast<float>(region.x + region.width),
           static_cast<float>(region.y + region.height), ret.classResult.second,
           static_cast<float>(ret.classResult.first)}};
      buf.algorithmResult.bboxes.emplace_back(std::move(b));
    } else if (type == "AlgorithmMessage") {
      if (buf.algorithmResult.bboxes.size() < 0) {
        return;
      }
      for (int i = 0; i < buf.algorithmResult.bboxes.size(); i++) {
        auto &bbox = buf.algorithmResult.bboxes.at(i);

        if (bbox.first == send) {
          cv::Rect rect{static_cast<int>(bbox.second[0]),
                        static_cast<int>(bbox.second[1]),
                        static_cast<int>(bbox.second[2] - bbox.second[0]),
                        static_cast<int>(bbox.second[3] - bbox.second[1])};
          if (rect.area() < 3 * 3) {
            continue;
          }
          cv::Mat croppedImage = (*image)(rect).clone();

          cv::cvtColor(croppedImage, croppedImage, cv::COLOR_RGB2BGR);
          cv::imwrite("/home/wangxt/workspace/projects/flowengine/tests/data/out.jpg", croppedImage);
          infer::Result ret;

          ret.shape = {croppedImage.cols, croppedImage.rows, 3};

          if (!instance->infer(croppedImage.data, ret)) {
            continue;
          }
          std::pair<std::string, std::array<float, 6>> b = {
              name,
              {bbox.second[0], bbox.second[1], bbox.second[2], bbox.second[3],
               ret.classResult.second,
               static_cast<float>(ret.classResult.first)}};

          buf.algorithmResult.bboxes.emplace_back(b);
        }
      }
    }
    autoSend(buf);
  }
}
} // namespace module
