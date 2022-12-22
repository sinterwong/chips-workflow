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
#include <iostream>
#include <string>
#include <unordered_map>

#include "common/config.hpp"
#include "logger/logger.hpp"

#include "rapidjson/document.h"
#include "rapidjson/filewritestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

namespace utils {

using common::ModuleConfigure;
using common::ParamsConfig;

class ConfigParser {
private:
  // 配置参数类型
  std::unordered_map<std::string, common::ConfigType> typeMapping {
    std::make_pair("stream", common::ConfigType::Stream),
    std::make_pair("algorithm", common::ConfigType::Algorithm),
    std::make_pair("output", common::ConfigType::Output),
    std::make_pair("logic", common::ConfigType::Logic),
  };

public:
  // ConfigParser(std::string const &url_) : url(url_) {}
  bool
  parseConfig(const char *jsonstr,
              std::vector<std::vector<std::pair<ModuleConfigure, ParamsConfig>>>
                  &pipelinesConfigs);

  bool readFile(std::string const &filename, std::string &result);

  bool writeJson(std::string const &config, std::string const &outPath);
};

} // namespace utils
#endif
