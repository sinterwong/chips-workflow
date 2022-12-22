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
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

#include "backend.h"
#include "common/common.hpp"
#include "classifier.hpp"
#if (TARGET_PLATFORM == 0)
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#elif (TARGET_PLATFORM == 1)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
#endif
#include "module.hpp"
namespace module {
class ClassifierModule : public Module {
private:
  int count = 0;
  common::AlgorithmConfig params;
  std::shared_ptr<AlgoInference> instance;
  infer::ModelInfo modelInfo;
  std::shared_ptr<infer::vision::Classifier> classifier;

public:
  ClassifierModule(Backend *ptr, const std::string &initName,
                   const std::string &initType,
                   const common::AlgorithmConfig &_params);

  ~ClassifierModule();

  void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // __METAENGINE_CLASSIFIER_MODULE_H_
