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

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "logicModule.h"

namespace module {
class HelmetModule : public LogicModule {

public:
  HelmetModule(Backend *ptr,
                   const std::string &initName, 
                   const std::string &initType, 
                   const common::LogicConfig &logicConfig,
                   const std::vector<std::string> &recv = {},
                   const std::vector<std::string> &send = {},
                   const std::vector<std::string> &pool = {});
  ~HelmetModule() {}

  virtual void forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
                   message) override;
  
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
