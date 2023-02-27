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

#include "common/config.hpp"
#include "logger/logger.hpp"
#include "module.hpp"

namespace module {
using common::OutputConfig;

static size_t curl_callback(void *ptr, size_t size, size_t nmemb,
                            std::string *data) {
  data->append((char *)ptr, size * nmemb);
  return size * nmemb;
}

class OutputModule : public Module {
protected:
  common::OutputConfig config;

public:
  OutputModule(backend_ptr ptr, std::string const &name,
               std::string const &type, OutputConfig const &config_)
      : Module(ptr, name, type), config(config_) {}
  ~OutputModule() {}
};
} // namespace module
#endif // __METAENGINE_SEND_STATUS_OUTPUT_H_
