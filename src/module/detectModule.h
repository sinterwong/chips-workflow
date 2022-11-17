/**
 * @file detectModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-03
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_DETECT_MODULE_H_
#define __METAENGINE_DETECT_MODULE_H_

#include <memory>
#include <string>

#include "backend.h"
#include "common/common.hpp"
#include "detection.hpp"
#if (TARGET_PLATFORM == 0)
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#elif (TARGET_PLATFORM == 1)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
#elif (TARGET_PLATFORM == 2)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
#endif
#include "module.hpp"

namespace module {
class DetectModule : public Module {
private:
  int count = 0;
  common::AlgorithmConfig params;
  std::shared_ptr<AlgoInference> instance;
  infer::ModelInfo modelInfo;
  std::shared_ptr<infer::vision::Detection> detector;

public:
  DetectModule(Backend *ptr, const std::string &initName,
               const std::string &initType,
               const common::AlgorithmConfig &_params);

  ~DetectModule();

  virtual void forward(std::vector<forwardMessage> message) override;
};
} // namespace module
#endif // __METAENGINE_DETECT_MODULE_H_
