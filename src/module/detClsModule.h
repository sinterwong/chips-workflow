/**
 * @file detClsModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-09-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_GENERAL_MODULE_H_
#define __METAENGINE_GENERAL_MODULE_H_

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include <any>
#include <memory>
#include <vector>

using common::DetClsMonitor;
using common::ModuleConfig;

namespace module {
class DetClsModule : Module {

  std::unique_ptr<DetClsMonitor> config;

public:
  DetClsModule(backend_ptr ptr, std::string const &name,
               MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config =
        std::make_unique<DetClsMonitor>(*config_.getParams<DetClsMonitor>());
  }

  ~DetClsModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
