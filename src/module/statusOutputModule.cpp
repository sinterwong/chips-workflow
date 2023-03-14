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
#include "pipeline.hpp"
#include <fstream>

namespace module {

StatusOutputModule::StatusOutputModule(backend_ptr ptr, std::string const &name,
                                       MessageType const &type,
                                       ModuleConfig &config_)
    : OutputModule(ptr, name, type, std::move(config_)) {}

bool StatusOutputModule::postResult(std::string const &url,
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

  rapidjson::Document doc;
  doc.SetObject(); // key-value 相当与map

  rapidjson::Document::AllocatorType &allocator =
      doc.GetAllocator(); // 获取分配器

  // /*
  // Add member
  rapidjson::Value moduleName(statusInfo.moduleName.c_str(), allocator);
  doc.AddMember("name", moduleName, allocator);
  // --------------------
  doc.AddMember("state", statusInfo.status, allocator);
  // */

  rapidjson::StringBuffer buffer;
  // PrettyWriter是格式化的json，如果是Writer则是换行空格压缩后的json
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  std::string out = std::string(buffer.GetString());

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, out.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, out.length());

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

  CURLcode response = curl_easy_perform(curl);
  if (response) {
    FLOWENGINE_LOGGER_ERROR(
        "[StatusOutputModule]: curl_easy_perform error code: {}", response);
    return false;
  }

  // end of for
  curl_easy_cleanup(curl);
  return true;
}

void StatusOutputModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} StatusOutputModule module was done!", name);
      stopFlag.store(true);
      return;
    }
    if (buf.status == 2 || count++ >= 500) {
      StatusInfo statusInfo{send, buf.status};
      std::string response;
      if (!postResult(config->url, statusInfo, response)) {
        FLOWENGINE_LOGGER_ERROR("StatusOutputModule.forward: post result was "
                                "failed, please check!");
      }
      count = 0;
    }
  }
}
FlowEngineModuleRegister(StatusOutputModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module