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

namespace utils {
bool ConfigParser::parseConfig(const char *jsonstr,
                               std::vector<common::FlowConfigure> &result) {
  // rapidjson::Document d;
  // if (d.Parse(jsonstr).HasParseError()) {
  //   // FLOWENGINE_LOGGER_ERROR("parseJson: parse error!");
  //   return false;
  // }
  // if (!d.IsObject()) {
  //   FLOWENGINE_LOGGER_ERROR("parseJson: should be an object!");
  //   return false;
  // }
  // if (d.HasMember("code")) {
  //   rapidjson::Value &m = d["code"];
  //   int code = m.GetInt();
  //   if (code != 200) {
  //     FLOWENGINE_LOGGER_ERROR("parseJson: failed response code {}", code);
  //     return false;
  //   }
  // }
  // FLOWENGINE_LOGGER_INFO("parseJson: Loading configure of camera....");
  // if (d.HasMember("data")) {
  //   rapidjson::Value &m = d["data"];
  //   if (m.IsArray()) {
  //     for (int i = 0; i < m.Size(); i++) {
  //       common::FlowConfigure fc;
  //       rapidjson::Value &e = m[i];
  //       if (e.HasMember("CameraIp")) {
  //         fc.status = e["Status"].GetInt();
  //         fc.hostId = e["HostId"].GetInt();
  //         fc.cameraId = e["CameraId"].GetInt();
  //         fc.provinceId = e["ProvinceId"].GetInt();
  //         fc.cityId = e["CityId"].GetInt();
  //         fc.regionId = e["RegionId"].GetInt();
  //         fc.stationId = e["StationId"].GetInt();
  //         fc.width = e["Width"].GetInt();
  //         fc.height = e["Height"].GetInt();
  //         fc.location = e["Location"].GetInt();
  //         fc.alarmType = e["AlarmType"].GetString();
  //         fc.videoCode = e["VideoCode"].GetString();
  //         fc.flowType = e["FlowType"].GetString();
  //         fc.cameraIp = e["CameraIp"].GetString();
  //         // std::string paramsConfigs = e["Config"].GetString();
  //         // 算法参数
  //         if (e.HasMember("Config") && e.IsObject()) {
  //           rapidjson::Value &c = e["Config"];
  //           if (c.HasMember("region")) {
  //             rapidjson::Value &r = c["region"];
  //             if (r.IsArray() && r.Size() == 4) {
  //               for (int i = 0; i < r.Size(); i++) {
  //                 rapidjson::Value &c = r[i];
  //                 fc.paramsConfig.algorithmConfig.region[i] = c.GetInt();
  //               }
  //             }
  //           }
  //           if (c.HasMember("nms_thr")) {
  //             fc.paramsConfig.algorithmConfig.nms_thr = c["nms_thr"].GetFloat();
  //           }
  //           if (c.HasMember("cond_thr")) {
  //             fc.paramsConfig.algorithmConfig.cond_thr = c["cond_thr"].GetFloat();
  //           }
  //           if (c.HasMember("modelDir")) {
  //             fc.paramsConfig.algorithmConfig.modelPath = c["modelDir"].GetString();
  //           }
  //         }
  //       }
  //       result.push_back(fc);
  //     }
  //   }
  // }
  return true;
}

bool ConfigParser::readFile(std::string const &filename, std::string &result) {
  // FILE *fp = fopen(filename.c_str(), "rb");
  // if (!fp) {
  //   FLOWENGINE_LOGGER_ERROR("open failed! file: {}", filename);
  //   return false;
  // }

  // char *buf = new char[1024 * 16];
  // int n = fread(buf, 1, 1024 * 16, fp);
  // fclose(fp);

  // if (n >= 0) {
  //   result.append(buf, 0, n);
  // }
  // delete[] buf;
  // std::string temp = "{\"code\": 200,\"msg\": \"SUCCESS\"}";
  // writeJson(temp, filename);
  return true;
}

bool ConfigParser::writeJson(std::string const &jsonstr,
                             std::string const &outPath) {
  // rapidjson::Document doc;
  // FLOWENGINE_LOGGER_INFO("writing json....");
  // FILE *fp = fopen(outPath.c_str(), "wb"); // non-Windows use "w"
  // char writeBuffer[65536];
  // rapidjson::FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));
  // rapidjson::PrettyWriter<rapidjson::FileWriteStream> writer(os);
  // if (doc.Parse(jsonstr.c_str()).HasParseError()) {
  //   FLOWENGINE_LOGGER_ERROR("parseJson: parse error!");
  //   return false;
  // }
  // doc.Accept(writer);
  // fclose(fp);
  return true;
}

// Main func begin
bool ConfigParser::postConfig(std::string const &url, int deviceId,
                              std::string &result) {

  // CURL *curl = curl_easy_init();

  // struct curl_slist *headers = NULL;
  // // without this 500 error
  // headers =
  //     curl_slist_append(headers, "Content-Type:application/json;charset=UTF-8");
  // curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  // curl_easy_setopt(curl, CURLOPT_POST, 1); //设置为非0表示本次操作为POST
  // curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

  // rapidjson::Document doc;
  // doc.SetObject(); // key-value 相当与map

  // rapidjson::Document::AllocatorType &allocator =
  //     doc.GetAllocator(); //获取分配器

  // // Add member
  // doc.AddMember("id", deviceId, allocator);

  // rapidjson::StringBuffer buffer;
  // // PrettyWriter是格式化的json，如果是Writer则是换行空格压缩后的json
  // rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  // doc.Accept(writer);

  // std::string s_out3 = std::string(buffer.GetString());

  // curl_easy_setopt(curl, CURLOPT_POSTFIELDS, s_out3.c_str());
  // curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, s_out3.length());

  // curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_callback);
  // curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

  // CURLcode response = curl_easy_perform(curl);

  // // end of for
  // curl_easy_cleanup(curl);
  return true;
}
} // namespace utils