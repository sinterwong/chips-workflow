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

#include <array>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

#include "backend.h"
#include "common/common.hpp"
#include "detection.hpp"
#include "yoloDet.hpp"
#if TARGET_PLATFORM == x3
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#elif TARGET_PLATFORM == jetson
#include "jetson/trt_inference.hpp"
using namespace infer::jetson;
#endif
#include "module.hpp"

namespace module {
class DetectModule : public Module {
private:
  int count = 0;
  // cv::Rect region{0, 0, 0, 0};
  common::AlgorithmConfig params;
  std::shared_ptr<AlgoInference> instance;
  std::shared_ptr<infer::vision::Detection> detector;

public:
  DetectModule(Backend *ptr, const std::string &initName,
               const std::string &initType,
               const common::AlgorithmConfig &_params,
               const std::vector<std::string> &recv = {},
               const std::vector<std::string> &send = {});

  ~DetectModule();

  virtual void forward(std::vector<forwardMessage> message) override;
};
} // namespace module
#endif // __METAENGINE_DETECT_MODULE_H_
