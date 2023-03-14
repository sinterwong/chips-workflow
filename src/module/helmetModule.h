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

#ifndef __METAENGINE_HELMET_MODULE_H_
#define __METAENGINE_HELMET_MODULE_H_
#include <any>
#include <memory>
#include <vector>
#include "logger/logger.hpp"
#include "module.hpp"
#include "common/common.hpp"

using common::WithoutHelmet;
using common::ModuleConfig;

namespace module {
class HelmetModule : Module {

  std::unique_ptr<WithoutHelmet> config;

public:
  HelmetModule(backend_ptr ptr, std::string const &name,
               MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config = std::unique_ptr<WithoutHelmet>(config_.getParams<WithoutHelmet>());
  }

  ~HelmetModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
