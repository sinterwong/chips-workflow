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

#ifndef __METAENGINE_SEND_STATUS_OUTPUT_H_
#define __METAENGINE_SEND_STATUS_OUTPUT_H_

#include <any>
#include <curl/curl.h>
#include <memory>
#include <vector>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "outputModule.h"

namespace module {

struct StatusInfo {
  std::string moduleName;
  int status;
};

class StatusOutputModule : public OutputModule {
private:
  int count = 0;

public:
  StatusOutputModule(backend_ptr ptr, std::string const &name,
                     MessageType const &type, ModuleConfig &config_)
      : OutputModule(ptr, name, type, config_) {}
  ~StatusOutputModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

  bool postResult(std::string const &url, StatusInfo const &resultInfo,
                  std::string &result);
};
} // namespace module
#endif // __METAENGINE_SEND_STATUS_OUTPUT_H_
