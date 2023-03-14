/**
 * @file configParser.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-05-30
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "configParser.hpp"
#include "common/common.hpp"
#include "logger/logger.hpp"
#include <fstream>
#include <utility>
#include <vector>

using common::Shape;

namespace utils {

bool ConfigParser::parseConfig(const char *jsonstr,
                               std::vector<pipelineParams> &pipelinesConfigs) {
  rapidjson::Document d;
  if (d.Parse(jsonstr).HasParseError()) {
    FLOWENGINE_LOGGER_ERROR("parseJson: parse error: ", d.GetParseError());
    return false;
  }
  if (!d.IsObject()) {
    FLOWENGINE_LOGGER_ERROR("parseJson: should be an object!");
    return false;
  }
  if (d.HasMember("Pipelines")) {
    rapidjson::Value &pipelines = d["Pipelines"];
    if (!pipelines.IsArray()) {
      FLOWENGINE_LOGGER_INFO("parseJson: Pipelines if not an array....");
      return false;
    }

    for (int i = 0; i < static_cast<int>(pipelines.Size()); i++) {
      rapidjson::Value &e = pipelines[i];
      if (!e.IsObject()) {
        continue;
      }
      rapidjson::Value &pipeline = e["Pipeline"];
      if (!pipeline.IsArray()) {
        FLOWENGINE_LOGGER_INFO("parseJson: Pipeline if not an array....");
        return false;
      }
      pipelineParams pipe;
      for (int p = 0; p < static_cast<int>(pipeline.Size()); p++) {

        rapidjson::Value &params = pipeline[p];
        if (!params.IsObject()) {
          continue;
        }
        if (!params.HasMember("type")) {
          FLOWENGINE_LOGGER_ERROR(
              "parseJson: Params doesn't contain type, please check!");
          return false;
        }
        std::string type = params["type"].GetString();
        common::ConfigType type_ = typeMapping.at(type);
        ParamsConfig pc;

        ModuleConfigure mc{
            params["sub_type"].GetString(), type, params["name"].GetString(),
            params["sendName"].GetString(), params["recvName"].GetString()};
        switch (type_) {
        case common::ConfigType::Stream: { // Stream
          pc = common::CameraConfig{
              params["name"].GetString(),     params["videoCode"].GetString(),
              params["flowType"].GetString(), params["cameraIp"].GetString(),
              params["width"].GetInt(),       params["height"].GetInt(),
              params["Id"].GetInt()};
          break;
        }
        case common::ConfigType::Algorithm: { // Algorithm
          rapidjson::Value &in = params["inputNames"];
          std::vector<std::string> inputNames;
          for (int i = 0; i < static_cast<int>(in.Size()); i++) {
            rapidjson::Value &n = in[i];
            inputNames.emplace_back(n.GetString());
          }

          rapidjson::Value &out = params["outputNames"];
          std::vector<std::string> outputNames;
          for (int i = 0; i < static_cast<int>(out.Size()); i++) {
            rapidjson::Value &n = out[i];
            outputNames.emplace_back(n.GetString());
          }

          rapidjson::Value &ins = params["inputShape"];
          Shape inputShape{ins[0].GetInt(), ins[1].GetInt(), ins[2].GetInt()};

          pc = common::AlgorithmConfig{
              params["modelPath"].GetString(),
              std::move(inputNames),
              std::move(outputNames),
              std::move(inputShape),
              params["algo_serial"].GetString(),
              params["cond_thr"].GetFloat(),
              params["nms_thr"].GetFloat(),
              params["alpha"].GetFloat(),
              params["beta"].GetFloat(),
              params["isScale"].GetBool(),
              params["batchSize"].GetInt(),
          };
          break;
        }
        case common::ConfigType::Logic: { // Logic
          std::array<int, 4> region{0, 0, 0, 0};
          if (params.HasMember("region")) {
            rapidjson::Value &rg = params["region"];
            region[0] = rg[0].GetInt();
            region[1] = rg[1].GetInt();
            region[2] = rg[2].GetInt();
            region[3] = rg[3].GetInt();
          }
          std::vector<int> attentionClasses;
          if (params.HasMember("attentionClasses")) {
            rapidjson::Value &ac = params["attentionClasses"];
            for (int i = 0; i < static_cast<int>(ac.Size()); i++) {
              rapidjson::Value &n = ac[i];
              attentionClasses.emplace_back(n.GetInt());
            }
          }

          pc = common::LogicConfig{params["alarm_output_dir"].GetString(),
                                   std::move(region),
                                   std::move(attentionClasses),
                                   params["event_id"].GetInt(),
                                   params["page"].GetString(),
                                   params["video_duration"].GetInt()};
          break;
        }
        case common::ConfigType::Output: { // Output
          pc = common::OutputConfig{params["url"].GetString()};
          break;
        }
        default: {
          break;
        }
        }
        pipe.emplace_back(moduleParams{std::move(mc), std::move(pc)});
      }
      if (!pipe.empty()) {
        pipelinesConfigs.emplace_back(pipe);
      }
    }
  }
  return true;
}

bool ConfigParser::readFile(std::string const &filename, std::string &result) {
  std::ifstream input_file(filename);
  if (!input_file.is_open()) {
    FLOWENGINE_LOGGER_INFO("Could not open the file - '{}'", filename);
    return false;
  }
  result = std::string((std::istreambuf_iterator<char>(input_file)),
                       std::istreambuf_iterator<char>());
  return true;
}

bool ConfigParser::writeJson(std::string const &jsonstr,
                             std::string const &outPath) {
  rapidjson::Document doc;
  FLOWENGINE_LOGGER_INFO("writing json....");
  FILE *fp = fopen(outPath.c_str(), "wb"); // non-Windows use "w"
  char writeBuffer[65536];
  rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
  rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(os);
  if (doc.Parse(jsonstr.c_str()).HasParseError()) {
    FLOWENGINE_LOGGER_ERROR("parseJson: parse error!");
    return false;
  }
  doc.Accept(writer);
  fclose(fp);
  return true;
}

} // namespace utils