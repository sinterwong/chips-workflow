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
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

#include "common/common.hpp"
#include "backend.h"
#include "frameMessage.pb.h"
#include "inference.h"
#include "jetson/detection.hpp"
#include "module.hpp"

namespace module {
class DetectModule : public Module {
private:
  int count = 0;
  cv::Rect region{0, 0, 0, 0};
  common::AlgorithmConfig params;
  std::shared_ptr<infer::trt::DetctionInfer> instance;

public:
  DetectModule(Backend *ptr, const std::string &initName,
               const std::string &initType, const common::AlgorithmConfig &_params,
               const std::vector<std::string> &recv = {},
               const std::vector<std::string> &send = {},
               const std::vector<std::string> &pool = {});

  ~DetectModule();

  void forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
                   message) override;
};
} // namespace module
#endif // __METAENGINE_DETECT_MODULE_H_
