/**
 * @file alarmOutputModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "alarmOutputModule.h"
#include "logger/logger.hpp"
#include "messageBus.h"
#include "outputModule.h"
#include <fstream>
#include <opencv2/imgcodecs.hpp>

namespace module {

AlarmOutputModule::AlarmOutputModule(backend_ptr ptr, std::string const &name,
                                     MessageType const &type,
                                     ModuleConfig &config_)
    : OutputModule(ptr, name, type, std::move(config_)) {}

bool AlarmOutputModule::postResult(std::string const &url,
                                   AlarmInfo const &alarmInfo,
                                   std::string &result) {

  CURL *curl = curl_easy_init();

  struct curl_slist *headers = NULL;
  // without this 500 error
  headers =
      curl_slist_append(headers, "Content-Type:application/json;charset=UTF-8");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1); // 设置为非0表示本次操作为POST
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

  rapidjson::Document doc;
  doc.SetObject(); // key-value 相当与map

  rapidjson::Document::AllocatorType &allocator =
      doc.GetAllocator(); // 获取分配器

  // /*
  // Add member
  rapidjson::Value alarm_type(alarmInfo.alarmType.c_str(), allocator);
  rapidjson::Value alarm_file(alarmInfo.alarmFile.c_str(), allocator);
  rapidjson::Value alarm_id(alarmInfo.alarmId.c_str(), allocator);
  rapidjson::Value alarm_detail(alarmInfo.alarmDetails.c_str(), allocator);
  rapidjson::Value camera_ip(alarmInfo.cameraIp.c_str(), allocator);
  rapidjson::Value page(alarmInfo.page.c_str(), allocator);
  rapidjson::Value algorithm_results(alarmInfo.algorithmResult.c_str(),
                                     allocator);
  doc.AddMember("alarm_file", alarm_file, allocator);
  doc.AddMember("alarm_type", alarm_type, allocator);
  doc.AddMember("alarm_id", alarm_id, allocator);
  doc.AddMember("alarm_detail", alarm_detail, allocator);
  doc.AddMember("camera_ip", camera_ip, allocator);
  doc.AddMember("page", page, allocator);
  doc.AddMember("algorithm_results", algorithm_results, allocator);
  // --------------------
  doc.AddMember("prepare_delay_in_sec", alarmInfo.prepareDelayInSec, allocator);
  doc.AddMember("event_id", alarmInfo.eventId, allocator);
  doc.AddMember("camera_id", alarmInfo.cameraId, allocator);
  doc.AddMember("width", alarmInfo.width, allocator);
  doc.AddMember("height", alarmInfo.height, allocator);
  // */

  rapidjson::StringBuffer buffer;
  // PrettyWriter是格式化的json，如果是Writer则是换行空格压缩后的json
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  std::string s_out3 = std::string(buffer.GetString());

  // std::ofstream out("/public/agent/out.json");
  // out << s_out3;
  // out.close();
  // exit(0);

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, s_out3.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, s_out3.length());

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

  CURLcode response = curl_easy_perform(curl);
  if (response) {
    FLOWENGINE_LOGGER_ERROR(
        "[AlarmOutputModule]: curl_easy_perform error code: {}", response);
    return false;
  }

  // end of for
  curl_easy_cleanup(curl);
  return true;
}

void AlarmOutputModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} AlarmOutputModule module was done!", name);
      stopFlag.store(true);
      return;
    }
    std::string response;
    if (!postResult(config->url, buf.alarmInfo, response)) {
      FLOWENGINE_LOGGER_ERROR(
          "AlarmOutputModule.forward: post result was failed, please check!");
    }
  }
}

FlowEngineModuleRegister(AlarmOutputModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module