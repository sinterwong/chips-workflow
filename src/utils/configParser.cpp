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
#include <utility>
#include <vector>

namespace utils {

bool ConfigParser::parseConfig(
    const char *jsonstr,
    std::vector<std::vector<std::pair<ModuleConfigure, ParamsConfig>>>
        &pipelinesConfigs) {
  rapidjson::Document d;
  if (d.Parse(jsonstr).HasParseError()) {
    // FLOWENGINE_LOGGER_ERROR("parseJson: parse error!");
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

    for (int i = 0; i < pipelines.Size(); i++) {
      rapidjson::Value &e = pipelines[i];
      if (!e.IsObject()) {
        continue;
      }
      rapidjson::Value &pipeline = e["Pipeline"];
      if (!pipeline.IsArray()) {
        FLOWENGINE_LOGGER_INFO("parseJson: Pipeline if not an array....");
        return false;
      }
      std::vector<std::pair<ModuleConfigure, ParamsConfig>> pipe;
      for (int p = 0; p < pipeline.Size(); p++) {

        rapidjson::Value &params = pipeline[p];
        if (!params.IsObject()) {
          continue;
        }
        if (!params.HasMember("type")) {
          FLOWENGINE_LOGGER_ERROR(
              "parseJson: Params doesn't contain type, please check!");
          return false;
        }
        common::ConfigType type_ = typeMapping.at(params["type"].GetString());
        std::string moduleType = params["sub_type"].GetString();
        ParamsConfig pc;
        ModuleConfigure mc{moduleMapping.at(moduleType),
                           params["type"].GetString(),
                           params["name"].GetString(),
                           params["sendName"].GetString(),
                           params["recvName"].GetString()};
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
          for (int i = 0; i < in.Size(); i++) {
            rapidjson::Value &n = in[i];
            inputNames.emplace_back(n.GetString());
          }

          rapidjson::Value &out = params["outputNames"];
          std::vector<std::string> outputNames;
          for (int i = 0; i < out.Size(); i++) {
            rapidjson::Value &n = out[i];
            outputNames.emplace_back(n.GetString());
          }

          // rapidjson::Value &ac = params["attentionClasses"];
          // std::vector<int> attentionClasses;
          // for (int i = 0; i < ac.Size(); i ++) {
          //   rapidjson::Value &n = ac[i];
          //   attentionClasses.emplace_back(n.GetString());
          // }
          // std::move(attentionClasses)

          rapidjson::Value &rg = params["region"];
          std::array<int, 4> region{rg[0].GetInt(), rg[1].GetInt(),
                                    rg[2].GetInt(), rg[3].GetInt()};

          rapidjson::Value &ins = params["inputShape"];
          std::array<int, 3> inputShape{ins[0].GetInt(), ins[1].GetInt(),
                                        ins[2].GetInt()};

          pc = common::AlgorithmConfig{
              params["modelPath"].GetString(),
              std::move(inputNames),
              std::move(outputNames),
              std::move(inputShape),
              params["numClasses"].GetInt(),
              params["numAnchors"].GetInt(),
              std::move(region),
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
          pc = common::LogicConfig{params["alarm_output_dir"].GetString(),
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
        pipe.emplace_back(std::pair<ModuleConfigure, ParamsConfig>{
            std::move(mc), std::move(pc)});
      }
      if (!pipe.empty()) {
        pipelinesConfigs.emplace_back(pipe);
      }
    }
  }
  return true;
}

bool ConfigParser::readFile(std::string const &filename, std::string &result) {
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    FLOWENGINE_LOGGER_ERROR("open failed! file: {}", filename);
    return false;
  }

  char *buf = new char[1024 * 16];
  int n = fread(buf, 1, 1024 * 16, fp);
  fclose(fp);

  if (n >= 0) {
    result.append(buf, 0, n);
  }
  delete[] buf;
  // std::string temp = "{\"code\": 200,\"msg\": \"SUCCESS\"}";
  std::string temp = "{}";
  writeJson(temp, filename);
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