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

using common::AlgoConfig;
using common::ModuleConfig;
using common::ModuleInfo;

namespace module::utils {

using ModuleParams = std::pair<ModuleInfo, ModuleConfig>;
using PipelineParams = std::vector<ModuleParams>;
using AlgorithmParams = std::pair<std::string, AlgoConfig>;

/**
 * @brief 组件的类型
 *
 */
enum class ModuleType { Algorithm, Stream, Output, Logic };

enum class SupportedFunction {
  DetClsModule = 0,
  CharsRecognitionModule,
  ObjectCounterModule
};

class ConfigParser {
private:
  // 模块映射
  std::unordered_map<std::string, SupportedFunction> moduleMapping{
      std::make_pair("CharsRecognitionModule",
                     SupportedFunction::CharsRecognitionModule),
      std::make_pair("DetClsModule", SupportedFunction::DetClsModule),
      std::make_pair("ObjectCounterModule", SupportedFunction::ObjectCounterModule),
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
                   std::vector<PipelineParams> &pipelines,
                   std::vector<AlgorithmParams> &algorithms);
};

} // namespace module::utils
#endif
