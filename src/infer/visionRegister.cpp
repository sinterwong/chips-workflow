/**
 * @file visionRegister.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "visionRegister.hpp"

#include "assdDet.hpp"
#include "crnnRec.hpp"
#include "faceKeyPoints.hpp"
#include "faceNet.hpp"
#include "infer_common.hpp"
#include "softmaxCls.hpp"
#include "yoloDet.hpp"
#include "yolopDet.hpp"
#include "yolov8pDet.hpp"

#include "utils/factory.hpp"

using namespace utils;

namespace infer::vision {
VisionRegistrar::VisionRegistrar() {
  FlowEngineModuleRegister(Assd, AlgoConfig const &, ModelInfo const &);
  FlowEngineModuleRegister(CRNN, AlgoConfig const &, ModelInfo const &);
  FlowEngineModuleRegister(Softmax, AlgoConfig const &, ModelInfo const &);
  FlowEngineModuleRegister(FaceNet, AlgoConfig const &, ModelInfo const &);
  FlowEngineModuleRegister(Yolo, AlgoConfig const &, ModelInfo const &);
  FlowEngineModuleRegister(YoloPDet, AlgoConfig const &, ModelInfo const &);
  FlowEngineModuleRegister(Yolov8PDet, AlgoConfig const &, ModelInfo const &);
  FlowEngineModuleRegister(FaceKeyPoints, AlgoConfig const &,
                           ModelInfo const &);
}

} // namespace infer::vision