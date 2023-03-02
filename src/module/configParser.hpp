/**
 * @file configParser.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __FLOWENGINE_JSON_PARSER_H_
#define __FLOWENGINE_JSON_PARSER_H_
#include "common/common.hpp"
#include <unordered_map>

using common::ModuleConfig;
using common::ModuleInfo;

namespace module::utils {

using ModuleParams = std::pair<ModuleInfo, ModuleConfig>;
using PipelineParams = std::vector<ModuleParams>;

/**
 * @brief 组件的类型
 *
 */
enum class ModuleType { Algorithm, Stream, Output, Logic };

enum class SupportedFunction {
  HelmetModule = 0,
  CallingModule,
  SmokingModule,
};

class ConfigParser {
private:
  // 模块映射
  std::unordered_map<std::string, SupportedFunction> moduleMapping{
      std::make_pair("HelmetModule", SupportedFunction::HelmetModule),
      std::make_pair("CallingModule", SupportedFunction::CallingModule),
      std::make_pair("SmokingModule", SupportedFunction::SmokingModule),
  };

  // 配置参数类型
  std::unordered_map<std::string, ModuleType> typeMapping{
      std::make_pair("stream", ModuleType::Stream),
      std::make_pair("algorithm", ModuleType::Algorithm),
      std::make_pair("output", ModuleType::Output),
      std::make_pair("logic", ModuleType::Logic),
  };

public:
  bool parseConfig(std::string const &path,
                   std::vector<PipelineParams> &pipelines);

  bool readFile(std::string const &filename, std::string &ret);

  bool writeJson(std::string const &config, std::string const &outPath);
};

} // namespace module::utils
#endif
