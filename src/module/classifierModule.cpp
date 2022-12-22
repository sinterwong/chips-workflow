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

#include "classifierModule.h"
#include "backend.h"
#include "classifier.hpp"
#include "infer_utils.hpp"
#include "inference.h"
#include "logger/logger.hpp"
#include "opencv2/imgproc.hpp"
#include <cassert>
#include <cstddef>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>

namespace module {

ClassifierModule::ClassifierModule(Backend *ptr, const std::string &initName,
                                   const std::string &initType,
                                   const common::AlgorithmConfig &_params)
    : Module(ptr, initName, initType), params(_params) {

  instance = std::make_shared<AlgoInference>(params);
  instance->initialize();

  infer::ModelInfo modelInfo;
  instance->getModelInfo(modelInfo);
  classifier = std::make_shared<infer::vision::Classifier>(params, modelInfo);
}

ClassifierModule::~ClassifierModule() {}

void ClassifierModule::forward(std::vector<forwardMessage> &message) {
  if (!instance) {
    FLOWENGINE_LOGGER_INFO("ClassifierModule {} instance is not init!", name);
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      FLOWENGINE_LOGGER_INFO("{} ClassifierModule module was done!", name);
      // std::cout << name << "{} Classifier module was done!" << std::endl;
      stopFlag.store(true);
      return;
    }
    auto frameBufMessage = backendPtr->pool->read(buf.key);
    auto image =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

    if (type == "logic") {
      if (count++ < 5)
        return;
      count = 0;
      cv::Rect2i region{buf.logicInfo.region[0], buf.logicInfo.region[1],
                        buf.logicInfo.region[2] - buf.logicInfo.region[0],
                        buf.logicInfo.region[3] - buf.logicInfo.region[1]};
      cv::Mat inferImage;
      infer::Result ret;
      if (region.area() != 0) {
        infer::utils::cropImage(*image, inferImage, region, buf.frameType);
        ret.shape = {region.width, region.height, 3};
      } else {
        inferImage = image->clone();
        ret.shape = {buf.cameraResult.widthPixel, buf.cameraResult.heightPixel,
                     3};
      }
      void *outputs[modelInfo.output_count];
      void *output = reinterpret_cast<void *>(outputs);
      FrameInfo frame;
      frame.data = reinterpret_cast<void **>(&inferImage.data);
      frame.shape = ret.shape;
      if (!instance->infer(frame, &output)) {
        continue;
      }
      classifier->processOutput(&output, ret);
      retBox b = {name,
                  {static_cast<float>(region.x), static_cast<float>(region.y),
                   static_cast<float>(region.x + region.width),
                   static_cast<float>(region.y + region.height),
                   ret.classResult.second,
                   static_cast<float>(ret.classResult.first)}};
      buf.algorithmResult.bboxes.emplace_back(std::move(b));
    } else if (type == "algorithm") {
      for (size_t i = 0; i < buf.algorithmResult.bboxes.size(); i++) {
        auto &bbox = buf.algorithmResult.bboxes.at(i);

        if (bbox.first == send) {
          cv::Rect2i rect{static_cast<int>(bbox.second[0]),
                          static_cast<int>(bbox.second[1]),
                          static_cast<int>(bbox.second[2] - bbox.second[0]),
                          static_cast<int>(bbox.second[3] - bbox.second[1])};
          cv::Mat inferImage;
          infer::utils::cropImage(*image, inferImage, rect, buf.frameType, 0.5);
          // cv::imwrite("classmodule.jpg", inferImage);
          infer::Result ret;
          ret.shape = {rect.width, rect.height, 3};
          FrameInfo frame;
          frame.data = reinterpret_cast<void **>(&inferImage.data);
          frame.shape = ret.shape;
          void *outputs[modelInfo.output_count];
          void *output = reinterpret_cast<void *>(outputs);
          if (!instance->infer(frame, &output)) {
            continue;
          }
          classifier->processOutput(&output, ret);
          retBox b = {name,
                      {bbox.second[0], bbox.second[1], bbox.second[2],
                       bbox.second[3], ret.classResult.second,
                       static_cast<float>(ret.classResult.first)}};

          buf.algorithmResult.bboxes.emplace_back(b);
        }
      }
    }
    autoSend(buf);
  }
}
FlowEngineModuleRegister(ClassifierModule, Backend *, std::string const &,
                         std::string const &, common::AlgorithmConfig const &);
} // namespace module
