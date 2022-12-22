/**
 * @file fireModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __METAENGINE_FIRE_MODULE_H_
#define __METAENGINE_FIRE_MODULE_H_

#include <any>
#include <memory>
#include <vector>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "logicModule.h"

namespace module {
class FireModule : public LogicModule {

public:
  FireModule(Backend *ptr, const std::string &initName,
               const std::string &initType,
               const common::LogicConfig &logicConfig);
  ~FireModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
