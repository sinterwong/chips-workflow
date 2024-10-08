/**
 * @file alarmOutputModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-12
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_SEND_OUTPUT_H_
#define __METAENGINE_SEND_OUTPUT_H_

#include <any>
#include <curl/curl.h>
#include <memory>
#include <vector>

#include "messageBus.hpp"

#include "logger/logger.hpp"
#include "module.hpp"

#include "common/common.hpp"

namespace module {
using common::ModuleConfig;
using common::OutputBase;

class OutputModule : public Module {
protected:
  std::unique_ptr<OutputBase> config;

public:
  OutputModule(backend_ptr ptr, std::string const &name,
               MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type, *config_.getParams<OutputBase>()) {
    config = std::make_unique<OutputBase>(*config_.getParams<OutputBase>());
  }
  ~OutputModule() {}
};
} // namespace module
#endif // __METAENGINE_SEND_STATUS_OUTPUT_H_
