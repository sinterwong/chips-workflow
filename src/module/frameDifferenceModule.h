/**
 * @file frameDifferenceModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-15
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_FRAME_DIFFERENCE_MODULE_H
#define __METAENGINE_FRAME_DIFFERENCE_MODULE_H

#include "alarmUtils.h"
#include "frame_difference.h"
#include "module.hpp"

namespace module {
using common::ModuleConfig;
using common::DetClsMonitor;
class FrameDifferenceModule : public Module {

public:
  FrameDifferenceModule(backend_ptr ptr, std::string const &name,
                        MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config = std::make_unique<DetClsMonitor>(
        *config_.getParams<DetClsMonitor>());
  }

  ~FrameDifferenceModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

private:
  int count = 0;
  infer::solution::FrameDifference fd;
  AlarmUtils alarmUtils;
  std::unique_ptr<DetClsMonitor> config;
};
} // namespace module
#endif // __METAENGINE_FRAME_DIFFERENCE_MODULE_H
