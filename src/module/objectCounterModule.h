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

#include "alarmUtils.h"
#include "infer/tracker.h"
#include "module.hpp"
#include <memory>

using common::ModuleConfig;
using common::ObjectCounterConfig;
using infer::solution::DeepSortTracker;

namespace module {
class ObjectCounterModule : Module {

  std::unique_ptr<ObjectCounterConfig> config;

public:
  ObjectCounterModule(backend_ptr ptr, std::string const &name,
                      MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config = std::make_unique<ObjectCounterConfig>(
        *config_.getParams<ObjectCounterConfig>());

    deepsort = std::make_unique<DeepSortTracker>(0.2, 100);
  }

  ~ObjectCounterModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

private:
  AlarmUtils alarmUtils;
  std::unique_ptr<DeepSortTracker> deepsort;  // 跟踪器
  std::set<int> counter;  // 计数器
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
