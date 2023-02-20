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
#include <memory>
#include <opencv2/core/mat.hpp>
#include <variant>

namespace infer {
using common::InferResult;
using common::RetBox;
using common::RetPoly;
using utils::cropImage;

bool VisionInfer::init() {
  instance = std::make_shared<AlgoInference>(config);
  instance->initialize();

  instance->getModelInfo(modelInfo);

  vision = ObjectFactory::createObject<vision::Vision>(config.algorithmSerial,
                                                       config, modelInfo);
  if (!vision) {
    FLOWENGINE_LOGGER_ERROR("Error algorithm serial {}",
                            config.algorithmSerial);
    return false;
  }
  return true;
}

bool VisionInfer::infer(void *data, const InferParams &params,
                        InferResult &ret) {
  // TODO data 直接默认是shared_ptr<cv::Mat>
  std::shared_ptr<cv::Mat> image{reinterpret_cast<cv::Mat *>(data)};
  auto &bboxes = params.regions;
  for (auto &bbox : bboxes) {
    cv::Rect2i rect{static_cast<int>(bbox.second[0]),
                    static_cast<int>(bbox.second[1]),
                    static_cast<int>(bbox.second[2] - bbox.second[0]),
                    static_cast<int>(bbox.second[3] - bbox.second[1])};
    cv::Mat inferImage;
    if (!cropImage(*image, inferImage, rect, params.frameType,
                   params.cropScaling)) {
      FLOWENGINE_LOGGER_ERROR(
          "VisionInfer: cropImage is failed, rect is {},{},{},{}, "
          "but the video resolution is {}x{}",
          rect.x, rect.y, rect.width, rect.height, image->cols, image->rows);
      return false;
    }
    ret.shape = {rect.width, rect.height, 3};
    FrameInfo frame;
    frame.data = reinterpret_cast<void **>(&inferImage.data);
    frame.shape = ret.shape;
    void *outputs[modelInfo.output_count];
    void *output = reinterpret_cast<void *>(outputs);
    if (!instance->infer(frame, &output)) {
      continue;
    }
    vision->processOutput(&output, ret);
  }
  return true;
}

} // namespace infer