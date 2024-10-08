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
#include "logger/logger.hpp"
#include "preprocess.hpp"
#include "utils/factory.hpp"
#include "vision.hpp"
#include "visionRegister.hpp"
#include <cstdlib>
#include <memory>
#include <mutex>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <variant>

using namespace utils;

namespace infer {
using common::InferResult;
using common::RetPoly;

bool VisionInfer::init() {

  vision::VisionRegistrar::getInstance();

  std::string aserial;
  config.visitParams([this, &aserial](auto const &params) {
    aserial = params.serial;
    instance = std::make_shared<AlgoInference>(params);
    if (!instance->initialize()) {
      FLOWENGINE_LOGGER_ERROR("VisionInfer initialize is failed!");
      return;
    }
    instance->getModelInfo(modelInfo);
    AlgoConfig p;
    p.setParams(params);
    vision = ObjectFactory::createObject<vision::Vision>(params.serial, p,
                                                         modelInfo);
  });

  if (!vision) {
    FLOWENGINE_LOGGER_ERROR("Error algorithm serial {}", aserial);
    return false;
  }
  serialName = aserial;
  serial = common::algoSerialMapping.at(aserial);
  retType = common::serial2TypeMapping.at(serial);
  return true;
}

bool VisionInfer::infer(FrameInfo &frame, const InferParams &params,
                        InferResult &ret) {
  // 栈上指针，infer后指向推理引擎管理的内存，无需在此处释放
  void *outputs[modelInfo.output_count];
  ret.shape = frame.shape;
  /*
   * 前处理并发性优化：在vision中实现CPU版本前处理，提高并发性
   */
  cv::Mat inputImage;
  auto preprocess_start = std::chrono::high_resolution_clock::now();
  if (!vision->processInput(frame, inputImage)) {
    FLOWENGINE_LOGGER_ERROR("VisionInfer infer: failed to processInput!");
    return false;
  }
  auto preprocess_stop = std::chrono::high_resolution_clock::now();
  auto preprocess_duration =
      std::chrono::duration_cast<std::chrono::microseconds>(preprocess_stop -
                                                            preprocess_start);
  auto preprocess_cost =
      static_cast<double>(preprocess_duration.count()) / 1000;
  {
    std::lock_guard lk(m);

    auto infer_start = std::chrono::high_resolution_clock::now();
    // if (!instance->infer(frame, &output)) {
    //   FLOWENGINE_LOGGER_ERROR("VisionInfer infer: failed to infer!");
    //   return false;
    // }
    // 不包含前处理的版本
    if (!instance->infer(inputImage, outputs)) {
      FLOWENGINE_LOGGER_ERROR("VisionInfer infer: failed to infer!");
      return false;
    }
    auto infer_stop = std::chrono::high_resolution_clock::now();
    auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        infer_stop - infer_start);
    auto infer_cost = static_cast<double>(infer_duration.count()) / 1000;

    /*
     * TODO: 后处理并发性优化
     * 问题：因为平台api相关问题，output指向的buffer管理在inference类中，因此此处数据不安全，需要加锁。
     * 受限于这个问题，后处理的无法并发完成。（ps:
     * 通常后处理的速度往往慢于内存开辟和拷贝速度。但基本上后处理就一两毫秒也不至于。）
     * 思路：后续考虑不在通过inference管理output的内存。output在这里开辟空间，infer时拷贝结果进去，
     * 处理完成后在此处释放（反正后处理都是在CPU上完成）从而提高并发性
     */
    auto post_start = std::chrono::high_resolution_clock::now();
    if (!vision->processOutput(outputs, ret)) {
      FLOWENGINE_LOGGER_ERROR("VisionInfer infer: failed to processOutput!");
      return false;
    }
    auto post_stop = std::chrono::high_resolution_clock::now();
    auto post_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        post_stop - post_start);
    auto post_cost = static_cast<double>(post_duration.count()) / 1000;

    FLOWENGINE_LOGGER_DEBUG("{} preprocess time: {} ms, infer time: {} ms, "
                            "post process time: {} ms",
                            serialName, preprocess_cost, infer_cost, post_cost);
  }
  return true;
}

bool VisionInfer::destory() noexcept { return true; }
} // namespace infer