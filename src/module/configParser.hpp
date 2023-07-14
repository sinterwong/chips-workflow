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

enum class SupportedFunc {
  DetClsModule = 0,
  OCRModule,
  LicensePlateModule,
  ObjectCounterModule,
  ObjectNumberModule,
  FrameDifferenceModule,
};

class ConfigParser {
private:
  // 模块映射
  std::unordered_map<std::string, SupportedFunc> moduleMapping{
      std::make_pair("OCRModule", SupportedFunc::OCRModule),
      std::make_pair("DetClsModule", SupportedFunc::DetClsModule),
      std::make_pair("LicensePlateModule", SupportedFunc::LicensePlateModule),
      std::make_pair("ObjectCounterModule", SupportedFunc::ObjectCounterModule),
      std::make_pair("ObjectNumberModule", SupportedFunc::ObjectNumberModule),
      std::make_pair("FrameDifferenceModule", SupportedFunc::FrameDifferenceModule),
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
