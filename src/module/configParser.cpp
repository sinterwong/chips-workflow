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

#include "module_utils.hpp"
#include "nlohmann/json.hpp"

#include <cstddef>
#include <pstl/glue_algorithm_defs.h>
#include <string>
#include <utility>

using common::algo_pipelines;
using common::AlgoBase;
using common::AlgoConfig;
using common::AttentionArea;
using common::ClassAlgo;
using common::DetAlgo;
using common::ExtinguisherMonitor;
using common::InferInterval;
using common::LogicBase;
using common::OutputBase;
using common::Point;
using common::SmokingMonitor;
using common::StreamBase;
using common::WithoutHelmet;

using json = nlohmann::json;

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

  // 获取Pipelines数组
  json pipes = config["Pipelines"];

  // 遍历 JSON 数据，生成 PipelineParams 对象
  for (auto const &pipe : pipes) {
    // 一个pipeline的所有参数
    PipelineParams params;

    // TODO 目前算法和模块平级，需要在获取算法模块的同时构造pipeline信息
    algo_pipelines algoPipes;

    // 解析 pipeline 中的参数
    for (const auto &p : pipe["Pipeline"]) {
      // 反序列化ModuleInfo
      ModuleInfo info;
      info.moduleName = p["name"].get<std::string>();
      info.moduleType = p["type"].get<std::string>();
      info.className = p["sub_type"].get<std::string>();
      info.sendName = p["sendName"].get<std::string>();
      info.recvName = p["recvName"].get<std::string>();

      // 分别解析各种类型功能的参数，给到功能参数中心
      ModuleConfig config;
      ModuleType type = typeMapping[info.moduleType];
      switch (type) {
      case ModuleType::Stream: {
        StreamBase stream_config;
        stream_config.cameraName = info.moduleName;
        stream_config.uri = p["cameraIp"].get<std::string>();
        stream_config.videoCode = p["videoCode"].get<std::string>();
        stream_config.flowType = p["flowType"].get<std::string>();
        stream_config.cameraId = p["Id"].get<int>();
        stream_config.height = p["height"].get<int>();
        stream_config.width = p["width"].get<int>();
        config.setParams(std::move(stream_config));
        break;
      }
      case ModuleType::Output: {
        OutputBase output_config;
        output_config.url = p["url"].get<std::string>();
        config.setParams(output_config);
        break;
      }
      case ModuleType::Algorithm: {
        // TODO 只能先根据名字来判断是什么类型的算法。。。
        std::string algo_name;
        std::string &module_name = info.moduleName;
        std::size_t start_pos = module_name.find("_");
        if (start_pos != std::string::npos) {
          start_pos += 1;
          std::size_t end_pos = module_name.find("_", start_pos);
          if (end_pos != std::string::npos) {
            algo_name = module_name.substr(start_pos, end_pos - start_pos);
          }
        }
        // TODO 收集算法pipe
        algoPipes.emplace_back(
            std::make_pair(algoMapping.at(algo_name), module_name));
        AlgoBase algo_base;
        algo_base.modelPath = p["modelPath"].get<std::string>();
        algo_base.serial = p["algo_serial"].get<std::string>();
        algo_base.batchSize = p["batchSize"].get<int>();
        algo_base.isScale = p["isScale"].get<bool>();
        algo_base.alpha = p["alpha"].get<float>();
        algo_base.beta = p["beta"].get<float>();
        algo_base.inputShape = p["inputShape"].get<std::array<int, 3>>();
        algo_base.inputNames = p["inputNames"].get<std::vector<std::string>>();
        algo_base.outputNames =
            p["outputNames"].get<std::vector<std::string>>();

        AlgoConfig algo_config; // 算法参数中心
        auto algo_serial = algoSerialMapping.at(algo_base.serial);
        switch (algo_serial) {
        case common::AlgoSerial::Yolo:
        case common::AlgoSerial::Assd: {
          float cond_thr = p["cond_thr"].get<float>();
          float nms_thr = p["nms_thr"].get<float>();
          DetAlgo det_config{std::move(algo_base), cond_thr, nms_thr};
          algo_config.setParams(std::move(det_config));
          break;
        }
        case common::AlgoSerial::Softmax: {
          ClassAlgo cls_config{std::move(algo_base)};
          algo_config.setParams(std::move(cls_config));
          break;
        }
        }
        config.setParams(algo_config);
        break;
      }
      case ModuleType::Logic: {
        // 公共参数部分
        LogicBase base_config;
        base_config.outputDir = p["alarm_output_dir"].get<std::string>();
        base_config.eventId = p["event_id"].get<int>();
        base_config.page = p["page"].get<std::string>();
        base_config.threshold = p["threshold"].get<float>();
        base_config.videDuration = p["video_duration"].get<int>();
        base_config.isDraw = true;
        // base_config.algoPipelines = algoPipes;

        SupportedFunction func = moduleMapping[info.className];
        switch (func) {
        case SupportedFunction::HelmetModule: {
          // TODO 未来每个模块都可以有自己特定的超参数
          AttentionArea aarea;
          auto region = p["region"].get<std::vector<int>>();
          for (size_t i = 0; i < region.size(); i+=2) {
            aarea.region.push_back(Point{region.at(i), region.at(i + 1)});
          }
          InferInterval interval;
          WithoutHelmet config_{std::move(aarea), std::move(base_config),
                                std::move(interval)};
          config.setParams(std::move(config_));
          break;
        }
        case SupportedFunction::SmokingModule: {
          // TODO 未来每个模块都可以有自己特定的超参数
          AttentionArea aarea;
          auto region = p["region"].get<std::vector<int>>();
          for (size_t i = 0; i < region.size(); i+=2) {
            aarea.region.push_back(Point{region.at(i), region.at(i + 1)});
          }
          InferInterval interval;
          SmokingMonitor config_{std::move(aarea), std::move(base_config),
                                 std::move(interval)};
          config.setParams(std::move(config_));
          break;
        }
        case SupportedFunction::ExtinguisherMonitor:
          // TODO 未来每个模块都可以有自己特定的超参数
          AttentionArea aarea;
          auto region = p["region"].get<std::vector<int>>();
          for (size_t i = 0; i < region.size(); i+=2) {
            aarea.region.push_back(Point{region.at(i), region.at(i + 1)});
          }
          ExtinguisherMonitor config_{std::move(aarea), std::move(base_config)};
          config.setParams(std::move(config_));
          break;
        }
        break;
      }
      }
      params.emplace_back(ModuleParams{std::make_pair(info, config)});
    }

    // TODO 模块的收发发生了变化，这里也需要临时的处理
    // TODO 删除掉recv是算法类型的logic
    params.erase(std::remove_if(params.begin(), params.end(),
                                [](auto &p) {
                                  auto &recvName = p.first.recvName;
                                  std::string recvType =
                                      recvName.substr(0, recvName.find("_"));
                                  return p.first.moduleType == "logic" &&
                                         recvType == "algorithm";
                                }),
                 params.end());

    // 目前都是单线的算法，因此只需要logic的尾部就可以
    std::string outputName;
    for (auto &param : params) {
      if (param.first.moduleType == "output") {
        outputName = param.first.moduleName;
      }
    }

    for (auto &param : params) {
      // 这里只会查到一个logic，尾部的上面已经过滤掉了
      if (param.first.moduleType == "logic") {
        // algoPipes;
        param.first.sendName = std::move(outputName);

        auto ctype = moduleMapping[param.first.className];
        switch (ctype) {
        case SupportedFunction::HelmetModule: {
          auto p = param.second.getParams<WithoutHelmet>();
          p->algoPipelines = std::move(algoPipes);
          break;
        }
        case SupportedFunction::ExtinguisherMonitor: {
          auto p = param.second.getParams<ExtinguisherMonitor>();
          p->algoPipelines = std::move(algoPipes);
          break;
        }
        case SupportedFunction::SmokingModule: {
          auto p = param.second.getParams<SmokingMonitor>();
          p->algoPipelines = std::move(algoPipes);
          break;
        }
        }
      }
    }
    pipelines.push_back(params);
  }

  return true;
}

} // namespace module::utils