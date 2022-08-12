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

#include "messageBus.h"
#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "module.hpp"

namespace module {

static size_t curl_callback(void *ptr, size_t size, size_t nmemb, std::string *data) {
  data->append((char *)ptr, size * nmemb);
  return size * nmemb;
}

class OutputModule : public Module {
protected:
  common::OutputConfig config;

public:
  OutputModule(Backend *ptr, const std::string &initName,
               const std::string &initType,
               const common::OutputConfig &outputConfig_,
               const std::vector<std::string> &recv = {},
               const std::vector<std::string> &send = {},
               const std::vector<std::string> &pool = {})
      : Module(ptr, initName, initType, recv, send, pool),
        config(std::move(outputConfig_)) {}
  ~OutputModule() {}
};
} // namespace module
#endif // __METAENGINE_SEND_STATUS_OUTPUT_H_
