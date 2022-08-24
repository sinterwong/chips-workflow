/**
 * @file SmokingModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-06-05
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __METAENGINE_CALLING_MODULE_H_
#define __METAENGINE_CALLING_MODULE_H_

#include <any>
#include <memory>
#include <vector>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "logicModule.h"

namespace module {
class SmokingModule : public LogicModule {

public:
  SmokingModule(Backend *ptr,
                   const std::string &initName, 
                   const std::string &initType, 
                   const common::LogicConfig &logicConfig,
                   const std::vector<std::string> &recv = {},
                   const std::vector<std::string> &send = {},
                   const std::vector<std::string> &pool = {});
  ~SmokingModule() {}

  void forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
                   message) override;
  
};
} // namespace module
#endif // __METAENGINE_CALLING_MODULE_H_
