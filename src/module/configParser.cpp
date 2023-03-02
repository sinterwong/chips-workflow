/**
 * @file configParser.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "configParser.hpp"
#include "logger/logger.hpp"
#include "logicModule.h"

#include <cstddef>
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

using common::AlgoBase;
using common::AlgoConfig;
using common::AttentionArea;
using common::ClassAlgo;
using common::DetAlgo;
using common::ExtinguisherMonitor;
using common::InferInterval;
using common::LogicBase;
using common::OutputBase;
using common::SmokingMonitor;
using common::StreamBase;
using common::WithoutHelmet;

namespace module::utils {
bool ConfigParser::parseConfig(std::string const &path,
                               std::vector<PipelineParams> &pipelines) {
  // 读取 JSON 文件
  std::string fileContents;
  if (!readFile(path, fileContents)) {
    return false;
  }

  // 解析 JSON 数据
  json config;
  try {
    config = json::parse(fileContents);
  } catch (std::exception const &e) {
    FLOWENGINE_LOGGER_ERROR("Failed to parse JSON: {}", e.what());
    return false;
  }

  // 遍历 JSON 数据，生成 PipelineParams 对象
  for (auto const &pipeline : config["Pipelines"]) {
    PipelineParams params;
    // 解析 pipeline 中的参数
    for (size_t i = 0; i < pipeline.size(); i++) {
      auto p = pipeline.at(i);

      // 反序列化ModuleInfo
      ModuleInfo info;
      info.moduleName = p["name"].get<std::string>();
      info.moduleType = p["type"].get<std::string>();
      info.className = p["sub_type"].get<std::string>();
      info.sendName = p["sendName"].get<std::string>();
      info.recvName = p["recvName"].get<std::string>();

      // TODO 分别解析各种类型功能的参数，给到功能参数中心
      ModuleConfig config;
      ModuleType type = typeMapping[info.moduleType];
      switch (type) {
      case ModuleType::Stream: {
        StreamBase stream_config;
        config.setParams(stream_config);
        break;
      }
      case ModuleType::Output: {
        OutputBase output_config;
        config.setParams(output_config);
        break;
      }
      case ModuleType::Algorithm: {
        // TODO 补全此处代码
        AlgoBase algo_base;
        AlgoConfig algo_config;
        config.setParams(algo_config);
        break;
      }
      case ModuleType::Logic: {
        // TODO 补全此处代码
        LogicBase base_config;
        SupportedFunction func = moduleMapping[info.className];
        switch (func) {
        case SupportedFunction::HelmetModule: {
          WithoutHelmet config_{AttentionArea(), std::move(base_config),
                                InferInterval()};
          break;
        }
        case SupportedFunction::CallingModule: {
          break;
        }
        case SupportedFunction::SmokingModule: {
          break;
        }
        }
        break;
      }
      }
    }
    pipelines.push_back(params);
  }

  return true;
}

bool ConfigParser::readFile(std::string const &filename, std::string &ret) {
  std::ifstream input_file(filename);
  if (!input_file.is_open()) {
    FLOWENGINE_LOGGER_ERROR("Failed to open file: '{}'", filename);
    return false;
  }
  ret = std::string((std::istreambuf_iterator<char>(input_file)),
                    std::istreambuf_iterator<char>());
  return true;
}

bool ConfigParser::writeJson(std::string const &config,
                             std::string const &outPath) {

  FLOWENGINE_LOGGER_INFO("Writing json....");
  json j = json::parse(config);
  std::ofstream out(outPath);
  if (!out.is_open()) {
    FLOWENGINE_LOGGER_ERROR("Failed to open file: '{}'", outPath);
    return false;
  }
  out << j.dump(4);
  return true;
}

} // namespace module::utils