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

#include "module_utils.hpp"
#include "nlohmann/json.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

using common::AlarmBase;
using common::algo_pipelines;
using common::AlgoBase;
using common::AlgoConfig;
using common::AlgoParams;
using common::AttentionArea;
using common::ClassAlgo;
using common::DetAlgo;
using common::DetClsMonitor;
using common::FeatureAlgo;
using common::LogicBase;
using common::ObjectCounterConfig;
using common::OCRConfig;
using common::OutputBase;
using common::PointsDetAlgo;
using Point2i = common::Point<int>;
using common::Points2i;
using common::StreamBase;

using json = nlohmann::json;

#define EXTRACT_JSON_VALUE(json_obj, key, var)                                 \
  do {                                                                         \
    if ((json_obj).count(key) > 0) {                                           \
      try {                                                                    \
        var = (json_obj)[key].get<decltype(var)>();                            \
      } catch (std::exception const &e) {                                      \
        FLOWENGINE_LOGGER_ERROR(                                               \
            "ConfigParser: Paramer extracting \"{}\" was failed!", key);       \
        return false;                                                          \
      }                                                                        \
    } else {                                                                   \
      FLOWENGINE_LOGGER_ERROR("ConfigParser: \"{}\" doesn't exist.", key);     \
      return false;                                                            \
    }                                                                          \
  } while (0)

namespace module::utils {
bool ConfigParser::parseConfig(std::string const &path,
                               std::vector<PipelineParams> &pipelines,
                               std::vector<AlgorithmParams> &algorithms) {
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

  // 获取算法配置
  json algos = config["Algorithms"];
  for (auto const &algo : algos) {
    AlgoBase algo_base;
    EXTRACT_JSON_VALUE(algo, "modelPath", algo_base.modelPath);
    EXTRACT_JSON_VALUE(algo, "algo_serial", algo_base.serial);
    EXTRACT_JSON_VALUE(algo, "batchSize", algo_base.batchSize);
    EXTRACT_JSON_VALUE(algo, "isScale", algo_base.isScale);
    EXTRACT_JSON_VALUE(algo, "alpha", algo_base.alpha);
    EXTRACT_JSON_VALUE(algo, "beta", algo_base.beta);
    EXTRACT_JSON_VALUE(algo, "inputShape", algo_base.inputShape);
    EXTRACT_JSON_VALUE(algo, "inputNames", algo_base.inputNames);
    EXTRACT_JSON_VALUE(algo, "outputNames", algo_base.outputNames);
    EXTRACT_JSON_VALUE(algo, "cond_thr", algo_base.cond_thr);

    std::string name;
    EXTRACT_JSON_VALUE(algo, "name", name);

    AlgoConfig algo_config; // 算法参数中心
    auto algo_serial = common::algoSerialMapping.at(algo_base.serial);
    switch (algo_serial) {
    case common::AlgoSerial::Yolo:
    case common::AlgoSerial::Assd: {
      float nms_thr;
      EXTRACT_JSON_VALUE(algo, "nms_thr", nms_thr);
      DetAlgo det_config{std::move(algo_base), nms_thr};
      algo_config.setParams(std::move(det_config));
      break;
    }
    case common::AlgoSerial::CRNN:
    case common::AlgoSerial::Softmax: {
      ClassAlgo cls_config{std::move(algo_base)};
      algo_config.setParams(std::move(cls_config));
      break;
    }
    case common::AlgoSerial::FaceNet: {
      int dim;
      EXTRACT_JSON_VALUE(algo, "dim", dim);
      FeatureAlgo feature_config{std::move(algo_base), dim};
      algo_config.setParams(std::move(feature_config));
      break;
    }
    case common::AlgoSerial::YoloPDet: {
      int numPoints;
      EXTRACT_JSON_VALUE(algo, "num_points", numPoints);
      float nms_thr;
      EXTRACT_JSON_VALUE(algo, "nms_thr", nms_thr);
      PointsDetAlgo pdet_config{std::move(algo_base), numPoints, nms_thr};
      algo_config.setParams(std::move(pdet_config));
      break;
    }
    }
    algorithms.emplace_back(std::pair{std::move(name), std::move(algo_config)});
  }

  // 获取Pipelines数组
  json pipes = config["Pipelines"];

  // 遍历 JSON 数据，生成 PipelineParams 对象
  for (auto const &pipe : pipes) {
    // 一个pipeline的所有参数
    PipelineParams params;

    // 解析 pipeline 中的参数
    for (const auto &p : pipe["Pipeline"]) {
      // 反序列化ModuleInfo
      ModuleInfo info;
      EXTRACT_JSON_VALUE(p, "name", info.moduleName);
      EXTRACT_JSON_VALUE(p, "type", info.moduleType);
      EXTRACT_JSON_VALUE(p, "sub_type", info.className);
      EXTRACT_JSON_VALUE(p, "sendName", info.sendName);
      EXTRACT_JSON_VALUE(p, "recvName", info.recvName);

      // 分别解析各种类型功能的参数，给到功能参数中心
      ModuleConfig config;
      ModuleType type = typeMapping[info.moduleType];
      switch (type) {
      case ModuleType::Stream: {
        StreamBase stream_config;
        stream_config.cameraName = info.moduleName;
        EXTRACT_JSON_VALUE(p, "cameraIp", stream_config.uri);
        EXTRACT_JSON_VALUE(p, "videoCode", stream_config.videoCode);
        EXTRACT_JSON_VALUE(p, "flowType", stream_config.flowType);
        EXTRACT_JSON_VALUE(p, "Id", stream_config.cameraId);
        EXTRACT_JSON_VALUE(p, "height", stream_config.height);
        EXTRACT_JSON_VALUE(p, "width", stream_config.width);
        if (p.count("runTime") > 0) {
          try {
            stream_config.runTime =
                p["runTime"].get<decltype(stream_config.runTime)>();
          } catch (std::exception const &e) {
            FLOWENGINE_LOGGER_ERROR(
                "ConfigParser: Paramer extracting \"{}\" was failed!",
                "runTime");
            return false;
          }
        }
        // EXTRACT_JSON_VALUE(p, "runTime", stream_config.runTime);

        config.setParams(std::move(stream_config));
        break;
      }
      case ModuleType::Output: {
        OutputBase output_config;
        EXTRACT_JSON_VALUE(p, "url", output_config.url);
        config.setParams(output_config);
        break;
      }
      case ModuleType::Algorithm: {
        // TODO 未来单独的算法组件
        break;
      }
      case ModuleType::Logic: {
        // 公共参数部分 DL的执行逻辑，属于必要字段
        LogicBase lBase;
        EXTRACT_JSON_VALUE(p, "interval", lBase.interval);
        // 前端传来的interval以秒为单位，需要转换成毫秒
        lBase.interval *= 1000;

        json apipes = p["algo_pipe"];
        for (auto const &ap : apipes) {
          AlgoParams params; // 特定参数
          EXTRACT_JSON_VALUE(ap, "attention", params.attentions);
          EXTRACT_JSON_VALUE(ap, "basedNames", params.basedNames);
          EXTRACT_JSON_VALUE(ap, "cropScaling", params.cropScaling);

          std::string name;
          EXTRACT_JSON_VALUE(ap, "name", name);
          lBase.algoPipelines.emplace_back(
              std::pair{std::move(name), std::move(params)});
        }

        // 前端划定区域 目前来说一定会有这个字段
        AttentionArea aarea;
        auto regions = p["regions"];
        for (auto const &region : regions) {
          Points2i ret;
          if (region.size() != 4) {
            // TODO 暂时将所有形状处理成矩形
            std::vector<int> oNumbers;
            std::vector<int> eNumbers;
            // 分别提取奇数位和偶数位的元素
            for (std::size_t i = 0; i < region.size(); ++i) {
              if (i % 2 == 0) {
                eNumbers.push_back(region[i]);
              } else {
                oNumbers.push_back(region[i]);
              }
            }
            auto x1 = std::min_element(oNumbers.begin(), oNumbers.end());
            auto x2 = std::max_element(oNumbers.begin(), oNumbers.end());
            auto y1 = std::min_element(eNumbers.begin(), eNumbers.end());
            auto y2 = std::max_element(eNumbers.begin(), eNumbers.end());
            ret.push_back(Point2i{*x1, *y1});
            ret.push_back(Point2i{*x2, *y2});
          } else {
            for (size_t i = 0; i < region.size(); i += 2) {
              ret.push_back(Point2i{region.at(i), region.at(i + 1)});
            }
          }
          aarea.regions.emplace_back(ret);
        }

        // 报警配置获取 目前来说一定会有这些字段
        std::string outputDir, page;
        int videoDuration, eventId;
        EXTRACT_JSON_VALUE(p, "alarm_output_dir", outputDir);
        EXTRACT_JSON_VALUE(p, "video_duration", videoDuration);
        EXTRACT_JSON_VALUE(p, "event_id", eventId);
        EXTRACT_JSON_VALUE(p, "page", page);

        auto func = moduleMapping.at(info.className);
        switch (func) {
        case SupportedFunc::OCRModule:
        case SupportedFunc::LicensePlateModule: {
          // 前端划定区域
          std::string chars;
          EXTRACT_JSON_VALUE(p, "chars", chars);

          auto isDraw = true;
          AlarmBase aBase{eventId, page, std::move(outputDir), videoDuration,
                          isDraw};

          OCRConfig config_{std::move(aarea), std::move(lBase),
                            std::move(aBase), std::move(chars)};
          config.setParams(std::move(config_));
          break;
        }
        case SupportedFunc::FrameDifferenceModule:
        case SupportedFunc::DetClsModule: {
          // 报警配置获取
          float thre;
          size_t requireExistence;
          EXTRACT_JSON_VALUE(p, "threshold", thre);
          EXTRACT_JSON_VALUE(p, "requireExistence", requireExistence);

          auto isDraw = true;
          AlarmBase aBase{eventId, page, std::move(outputDir), videoDuration,
                          isDraw};

          DetClsMonitor config_{std::move(aarea), std::move(lBase),
                                std::move(aBase), thre, requireExistence};
          config.setParams(std::move(config_));
          break;
        }
        case SupportedFunc::ObjectCounterModule:
        case SupportedFunc::ObjectNumberModule: {
          int amount;
          EXTRACT_JSON_VALUE(p, "amount", amount);

          auto isDraw = true;
          AlarmBase aBase{eventId, page, std::move(outputDir), videoDuration,
                          isDraw};

          ObjectCounterConfig config_{std::move(aarea), std::move(lBase),
                                      std::move(aBase), amount};
          config.setParams(std::move(config_));
          break;
        }
        }
        break;
      }
      }
      params.emplace_back(ModuleParams{std::make_pair(info, config)});
    }
    pipelines.push_back(params);
  }
  return true;
}

} // namespace module::utils