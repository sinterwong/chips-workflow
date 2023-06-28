/**
 * @file statusOutputModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "statusOutputModule.h"
#include <chrono>
#include <fstream>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace module {

CURLcode StatusOutputModule::postResult(std::string const &url,
                                        StatusInfo const &statusInfo,
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
  info["name"] = statusInfo.moduleName;
  info["state"] = statusInfo.status;

  std::string out = info.dump();

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, out.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, out.length());

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

  CURLcode response = curl_easy_perform(curl);
  // end of for
  curl_easy_cleanup(curl);
  return response;
}

void StatusOutputModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} StatusOutputModule was done!", name);
      stopFlag.store(true);
      return;
    }
    if (buf.status == 2 || count++ >= 5) {
      StatusInfo statusInfo{send, buf.status};
      std::string response;
      auto code = postResult(config->url, statusInfo, response);
      if (code) {
        FLOWENGINE_LOGGER_ERROR("StatusOutputModule.forward: post result was "
                                "failed {}, please check!",
                                code);
      }
      count = 0;
    }
  }
  std::this_thread::sleep_for(std::chrono::seconds(1));
}
FlowEngineModuleRegister(StatusOutputModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module