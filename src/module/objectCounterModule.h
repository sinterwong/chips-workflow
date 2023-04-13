/**
 * @file objectCounterModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2023-04-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __METAENGINE_OBJECT_COUNTER_MODULE_H_
#define __METAENGINE_OBJECT_COUNTER_MODULE_H_

#include "module.hpp"

#include "alarmUtils.h"

using common::DetClsMonitor;
using common::ModuleConfig;

namespace module {
class ObjectCounterModule : Module {

  std::unique_ptr<DetClsMonitor> config;

public:
  ObjectCounterModule(backend_ptr ptr, std::string const &name,
               MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config =
        std::make_unique<DetClsMonitor>(*config_.getParams<DetClsMonitor>());
  }

  ~ObjectCounterModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

private:
  AlarmUtils alarmUtils;
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
