/**
 * @file helmetModule.h
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

using common::ModuleConfig;
using common::GeneralMonitor;

namespace module {
class GeneralModule : Module {

  std::unique_ptr<GeneralMonitor> config;

public:
  GeneralModule(backend_ptr ptr, std::string const &name,
                MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config = std::unique_ptr<GeneralMonitor>(config_.getParams<GeneralMonitor>());
  }

  ~GeneralModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
