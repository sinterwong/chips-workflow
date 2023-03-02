/**
 * @file visionInfer.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "visionInfer.hpp"
#include "core/factory.hpp"
#include "preprocess.hpp"
#include "vision.hpp"
#include <cstdlib>
#include <memory>
#include <mutex>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <variant>

namespace infer {
using common::InferResult;
using common::RetBox;
using common::RetPoly;
using utils::cropImage;

bool VisionInfer::init() {

  std::string serial;
  config.visitParams([this, &serial](auto &params) {
    serial = params.serial;
    instance = std::make_shared<AlgoInference>(params);
    if (!instance->initialize()) {
      FLOWENGINE_LOGGER_ERROR("VisionInfer initialize is failed!");
      return;
    }
    instance->getModelInfo(modelInfo);
    vision = ObjectFactory::createObject<vision::Vision>(params.serial, params,
                                                         modelInfo);
  });

  if (!vision) {
    FLOWENGINE_LOGGER_ERROR("Error algorithm serial {}", serial);
    return false;
  }
  return true;
}

bool VisionInfer::infer(void *data, const InferParams &params,
                        InferResult &ret) {
  // 该函数可能被并发调用，infer时处理线程安全问题
  Shape const &shape = params.shape;
  auto cv_type = shape.at(2) == 1 ? CV_8UC1 : CV_8UC3;
  cv::Mat image = cv::Mat(shape.at(1), shape.at(0), cv_type, data);
  auto &bboxes = params.regions;
  for (auto &bbox : bboxes) {
    cv::Mat inferImage;
    cv::Rect2i rect{static_cast<int>(bbox.second[0]),
                    static_cast<int>(bbox.second[1]),
                    static_cast<int>(bbox.second[2] - bbox.second[0]),
                    static_cast<int>(bbox.second[3] - bbox.second[1])};
    if (rect.area() != 0) {
      if (!cropImage(image, inferImage, rect, params.frameType,
                     params.cropScaling)) {
        FLOWENGINE_LOGGER_ERROR(
            "VisionInfer infer: cropImage is failed, rect is {},{},{},{}, "
            "but the video resolution is {}x{}",
            rect.x, rect.y, rect.width, rect.height, image.cols, image.rows);
        return false;
      }
    } else {
      inferImage = image.clone();
    }
    switch (params.frameType) {
    case common::ColorType::None:
    case common::ColorType::RGB888:
    case common::ColorType::BGR888:
      ret.shape = {inferImage.cols, inferImage.rows, 3};
      break;
    case common::ColorType::NV12:
      ret.shape = {inferImage.cols, inferImage.rows * 2 / 3, 3};
      break;
    }
    FrameInfo frame;
    frame.type = params.frameType;
    frame.data = reinterpret_cast<void **>(&inferImage.data);
    frame.shape = ret.shape;
    void *outputs[modelInfo.output_count];
    void *output = reinterpret_cast<void *>(outputs);
    { // 不可多线程一起使用一个模型推理，其余的部分都可以并行处理，因此在这里上锁
      std::lock_guard lk(m);
      if (!instance->infer(frame, &output)) {
        FLOWENGINE_LOGGER_ERROR("VisionInfer infer: failed to infer!");
        return false;
      }
    }
    if (!vision->processOutput(&output, ret)) {
      FLOWENGINE_LOGGER_ERROR("VisionInfer infer: failed to processOutput!");
      return false;
    }
  }
  return true;
}

bool VisionInfer::destory() {
  // TODO 如何在这里销毁算法，并让自己结束运行
  return true;
}

} // namespace infer