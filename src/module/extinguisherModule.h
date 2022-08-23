/**
 * @file extinguisherModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __METAENGINE_EXTINGUISHER_MODULE_H_
#define __METAENGINE_EXTINGUISHER_MODULE_H_

#include <any>
#include <memory>
#include <vector>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "logicModule.h"

namespace module {
class ExtinguisherModule : public LogicModule {

public:
  ExtinguisherModule(Backend *ptr,
                   const std::string &initName, 
                   const std::string &initType, 
                   const common::LogicConfig &logicConfig,
                   const std::vector<std::string> &recv = {},
                   const std::vector<std::string> &send = {},
                   const std::vector<std::string> &pool = {});
  ~ExtinguisherModule() {}

  void forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
                   message) override;
  
};
} // namespace module
#endif // __METAENGINE_EXTINGUISHER_MODULE_H_
