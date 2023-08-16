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
      var = (json_obj)[key].get<decltype(var)>();                              \
    } else {                                                                   \
      FLOWENGINE_LOGGER_ERROR("ConfigParser: {} doesn't exist.", key);         \
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

  try {
    // 获取算法配置
    json algos = config["Algorithms"];
    for (auto const &algo : algos) {
      AlgoBase algo_base;
      algo_base.modelPath = algo["modelPath"].get<std::string>();
      algo_base.serial = algo["algo_serial"].get<std::string>();
      algo_base.batchSize = algo["batchSize"].get<int>();
      algo_base.isScale = algo["isScale"].get<bool>();
      algo_base.alpha = algo["alpha"].get<float>();
      algo_base.beta = algo["beta"].get<float>();
      algo_base.inputShape = algo["inputShape"].get<std::array<int, 3>>();
      algo_base.inputNames = algo["inputNames"].get<std::vector<std::string>>();
      algo_base.outputNames =
          algo["outputNames"].get<std::vector<std::string>>();
      algo_base.cond_thr = algo["cond_thr"].get<float>();

      std::string name = algo["name"].get<std::string>();

      AlgoConfig algo_config; // 算法参数中心
      auto algo_serial = common::algoSerialMapping.at(algo_base.serial);
      switch (algo_serial) {
      case common::AlgoSerial::Yolo:
      case common::AlgoSerial::Assd: {
        float nms_thr = algo["nms_thr"].get<float>();
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
        int dim = algo["dim"].get<int>();
        FeatureAlgo feature_config{std::move(algo_base), dim};
        algo_config.setParams(std::move(feature_config));
        break;
      }
      case common::AlgoSerial::YoloPDet: {
        int numPoints = algo["num_points"].get<int>();
        float nms_thr = algo["nms_thr"].get<float>();
        PointsDetAlgo pdet_config{std::move(algo_base), numPoints, nms_thr};
        algo_config.setParams(std::move(pdet_config));
        break;
      }
      }
      algorithms.emplace_back(
          std::pair{std::move(name), std::move(algo_config)});
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
          // TODO 未来单独的算法组件
          break;
        }
        case ModuleType::Logic: {
          // 公共参数部分 DL的执行逻辑，属于必要字段
          LogicBase lBase;
          // 前端传来的interval以秒为单位，需要转换成毫秒
          lBase.interval = p["interval"].get<int>() * 1000;
          json apipes = p["algo_pipe"];
          for (auto const &ap : apipes) {
            AlgoParams params; // 特定参数
            params.attentions = ap["attention"].get<std::vector<int>>();
            params.basedNames =
                ap["basedNames"].get<std::vector<std::string>>();
            params.cropScaling = ap["cropScaling"].get<float>();
            lBase.algoPipelines.emplace_back(
                std::pair{ap["name"].get<std::string>(), std::move(params)});
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
          auto outputDir = p["alarm_output_dir"].get<std::string>();
          auto videoDuration = p["video_duration"].get<int>();
          auto eventId = p["event_id"].get<int>();
          auto page = p["page"].get<std::string>();

          auto func = moduleMapping.at(info.className);
          switch (func) {
          case SupportedFunc::OCRModule:
          case SupportedFunc::LicensePlateModule: {
            // 前端划定区域
            auto chars = p["chars"].get<std::string>();
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
            auto thre = p["threshold"].get<float>();
            auto requireExistence = p["requireExistence"].get<size_t>();
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
            auto amount = p["amount"].get<int>();
            auto isDraw = false;
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
  } catch (std::exception const &e) {
    FLOWENGINE_LOGGER_ERROR("Deserialization failed. {}", e.what());
    return false;
  }
  return true;
}

} // namespace module::utils