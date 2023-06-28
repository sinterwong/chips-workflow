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
#include <chrono>
#include <fstream>
#include <opencv2/imgcodecs.hpp>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace module {

CURLcode AlarmOutputModule::postResult(std::string const &url,
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

  json info;

  // Add member
  info["alarm_file"] = alarmInfo.alarmFile;
  info["alarm_type"] = alarmInfo.alarmType;
  info["alarm_id"] = alarmInfo.alarmId;
  info["alarm_detail"] = alarmInfo.alarmDetails;
  info["camera_ip"] = alarmInfo.cameraIp;
  info["page"] = alarmInfo.page;
  info["algorithm_results"] = alarmInfo.algorithmResult;
  info["prepare_delay_in_sec"] = alarmInfo.prepareDelayInSec;
  info["event_id"] = alarmInfo.eventId;
  info["camera_id"] = alarmInfo.cameraId;
  info["width"] = alarmInfo.width;
  info["height"] = alarmInfo.height;

  std::string s_out3 = info.dump();

  // std::ofstream out("/public/agent/out.json");
  // out << s_out3;
  // out.close();
  // exit(0);

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, s_out3.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, s_out3.length());

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

  CURLcode response = curl_easy_perform(curl);

  // end of for
  curl_easy_cleanup(curl);
  return response;
}

void AlarmOutputModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} AlarmOutputModule was done!", name);
      stopFlag.store(true);
      return;
    }
    std::string response;
    auto code = postResult(config->url, buf.alarmInfo, response);
    if (code) {
      FLOWENGINE_LOGGER_ERROR("AlarmOutputModule : post result was "
                              "failed {}, please check!",
                              code);
    }
  }
  std::this_thread::sleep_for(std::chrono::microseconds(300));
}

FlowEngineModuleRegister(AlarmOutputModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module