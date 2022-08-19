/**
 * @file classifierModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-07-18
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __METAENGINE_CLASSIFIER_MODULE_H_
#define __METAENGINE_CLASSIFIER_MODULE_H_

#include <array>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

#include "common/common.hpp"
#include "backend.h"
#include "frameMessage.pb.h"
#include "inference.h"
#include "jetson/classifier.hpp"
#include "module.hpp"

namespace module {
class ClassifierModule : public Module {
private:
  std::shared_ptr<infer::trt::ClassifierInfer> instance;
  int count = 0;
  common::AlgorithmConfig params;
  // cv::Rect region{0, 0, 0, 0};

public:
  ClassifierModule(Backend *ptr, const std::string &initName,
               const std::string &initType, const common::AlgorithmConfig &_params,
               const std::vector<std::string> &recv = {},
               const std::vector<std::string> &send = {},
               const std::vector<std::string> &pool = {});

  ~ClassifierModule();

  void forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
                   message) override;
};
} // namespace module
#endif // __METAENGINE_CLASSIFIER_MODULE_H_
