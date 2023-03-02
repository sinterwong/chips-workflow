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

#include "logicModule.h"
#include "module.hpp"

namespace module {
class HelmetModule : Module {

public:
  HelmetModule(backend_ptr ptr, std::string const &name,
               MessageType const &type, LogicBase const &logicConfig)
      : Module(ptr, name, type) {}

  ~HelmetModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
