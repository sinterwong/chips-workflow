/**
 * @file config.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-05-30
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __FLOWENGINE_PARSER_CONFIG_H_
#define __FLOWENGINE_PARSER_CONFIG_H_
#include <array>
#include <curl/curl.h>
#include <iostream>
#include <string>
#include <unordered_map>

#include "common/common.hpp"
#include "logger/logger.hpp"

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace utils {
static size_t curl_callback(void *ptr, size_t size, size_t nmemb,
                            std::string *data) {
  data->append((char *)ptr, size * nmemb);
  return size * nmemb;
}

class ConfigParser {
public:
  // parse json info to flowconfigure sturct
  bool parseConfig(char const *jsonstr, std::vector<common::FlowConfigure> &);

  bool parseParams(const char *jsonstr, common::ParamsConfig &result);

  bool readFile(std::string const &filename, std::string &result);

  bool writeJson(std::string const &config, std::string const &outPath);

  bool postConfig(std::string const &url, int deviceId, std::string &result);
};

} // namespace utils
#endif
