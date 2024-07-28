/**
 * @file alarmOutputModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_SEND_ALARM_OUTPUT_H_
#define __METAENGINE_SEND_ALARM_OUTPUT_H_

#include <any>
#include <curl/curl.h>
#include <memory>
#include <string>
#include <vector>

#include "messageBus.hpp"

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "outputModule.hpp"

namespace module {

class AlarmOutputModule : public OutputModule {

public:
  AlarmOutputModule(backend_ptr ptr, std::string const &name,
                    MessageType const &type, ModuleConfig &config_)
      : OutputModule(ptr, name, type, config_) {}
  ~AlarmOutputModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

  CURLcode postResult(std::string const &url, AlarmInfo const &resultInfo,
                      std::string &result);
};
} // namespace module
#endif // __METAENGINE_SEND_ALARM_OUTPUT_H_
