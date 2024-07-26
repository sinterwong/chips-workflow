/**
 * @file objectNumber.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-04-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __METAENGINE_OBJECT_NUMBER_MODULE_H_
#define __METAENGINE_OBJECT_NUMBER_MODULE_H_

#include "alarmUtils.hpp"
#include "module.hpp"
#include <memory>

using common::ModuleConfig;
using common::ObjectCounterConfig;

namespace module {
class ObjectNumberModule : Module {

  std::unique_ptr<ObjectCounterConfig> config;

public:
  ObjectNumberModule(backend_ptr ptr, std::string const &name,
                     MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type, *config_.getParams<ObjectCounterConfig>()) {
    config = std::make_unique<ObjectCounterConfig>(
        *config_.getParams<ObjectCounterConfig>());
  }

  ~ObjectNumberModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

private:
  AlarmUtils alarmUtils;
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
