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
using common::AlgoSerial;
using common::ModuleConfig;
using common::ModuleInfo;
using common::SupportedAlgo;

namespace module::utils {

using ModuleParams = std::pair<ModuleInfo, ModuleConfig>;
using PipelineParams = std::vector<ModuleParams>;
using AlgorithmParams = std::pair<std::string, AlgoConfig>;

/**
 * @brief 组件的类型
 *
 */
enum class ModuleType { Algorithm, Stream, Output, Logic };

enum class SupportedFunction { HelmetModule = 0, GeneralModule };

class ConfigParser {
private:
  // 模块映射
  std::unordered_map<std::string, SupportedFunction> moduleMapping{
      std::make_pair("HelmetModule", SupportedFunction::HelmetModule),
      std::make_pair("GeneralModule", SupportedFunction::GeneralModule),
  };

  // 配置参数类型
  std::unordered_map<std::string, ModuleType> typeMapping{
      std::make_pair("stream", ModuleType::Stream),
      std::make_pair("algorithm", ModuleType::Algorithm),
      std::make_pair("output", ModuleType::Output),
      std::make_pair("logic", ModuleType::Logic),
  };

  // 算法系列映射
  std::unordered_map<std::string, AlgoSerial> algoSerialMapping{
      std::make_pair("Yolo", AlgoSerial::Yolo),
      std::make_pair("Assd", AlgoSerial::Assd),
      std::make_pair("Softmax", AlgoSerial::Softmax),
  };

  // TODO 支持的算法功能映射
  std::unordered_map<std::string, SupportedAlgo> algoMapping{
      std::make_pair("handDet", SupportedAlgo::HandDet),
      std::make_pair("headDet", SupportedAlgo::HeadDet),
      std::make_pair("phoneCls", SupportedAlgo::SmokeCallCls),
      std::make_pair("helmetCls", SupportedAlgo::HelmetCls),
      std::make_pair("smokeCls", SupportedAlgo::SmokeCallCls),
      std::make_pair("extinguisherCls", SupportedAlgo::ExtinguisherCls),
  };

public:
  bool parseConfig(std::string const &path,
                   std::vector<PipelineParams> &pipelines,
                   std::vector<AlgorithmParams> &algorithms);
};

} // namespace module::utils
#endif
