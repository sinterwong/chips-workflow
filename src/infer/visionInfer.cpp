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

bool VisionInfer::init() {

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
  serial = common::algoSerialMapping.at(aserial);
  retType = common::serial2TypeMapping.at(serial);
  return true;
}

bool VisionInfer::infer(FrameInfo &frame, const InferParams &params,
                        InferResult &ret) {

  void *outputs[modelInfo.output_count];
  void *output = reinterpret_cast<void *>(outputs);
  ret.shape = frame.shape;
  {
    /*
     * TODO: 前处理并发性优化
     * 问题：之前考虑各个平台有相应的图像处理加速模块所以将前处理包含在infer中，但是目前有部分芯片的前处理依然使用cpu完成，
     * 所以此处前处理的并发性有优化空间。
     * 思路：后续考虑在vision中实现CPU版本前处理，再配合配置参数决定是否使用vision中的前处理，从而提高并发性
     */
    std::lock_guard lk(m);
    if (!instance->infer(frame, &output)) {
      FLOWENGINE_LOGGER_ERROR("VisionInfer infer: failed to infer!");
      return false;
    }
    /*
     * TODO: 后处理并发性优化
     * 问题：因为平台api相关问题，output指向的buffer管理在inference类中，因此此处数据不安全，需要加锁。
     * 受限于这个问题，后处理的无法并发完成。（ps:
     * 通常后处理的速度往往慢于内存开辟和拷贝速度。）
     * 思路：后续考虑不在通过inference管理output的内存。output在这里开辟空间，infer时拷贝结果进去，
     * 处理完成后在此处释放（反正后处理都是在CPU上完成）从而提高并发性
     */
    if (!vision->processOutput(&output, ret)) {
      FLOWENGINE_LOGGER_ERROR("VisionInfer infer: failed to processOutput!");
      return false;
    }
  }

  return true;
}

bool VisionInfer::destory() noexcept { return true; }

} // namespace infer