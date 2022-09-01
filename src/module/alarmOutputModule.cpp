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
#include "messageBus.h"
#include "outputModule.h"
#include <fstream>
#include <opencv2/imgcodecs.hpp>

namespace module {

AlarmOutputModule::AlarmOutputModule(Backend *ptr, const std::string &initName,
                                     const std::string &initType,
                                     const common::OutputConfig &outputConfig,
                                     const std::vector<std::string> &recv,
                                     const std::vector<std::string> &send)
    : OutputModule(ptr, initName, initType, outputConfig, recv, send) {}

bool AlarmOutputModule::postResult(std::string const &url,
                                   AlarmInfo const &alarmInfo,
                                   std::string &result) {

  CURL *curl = curl_easy_init();

  struct curl_slist *headers = NULL;
  // without this 500 error
  headers =
      curl_slist_append(headers, "Content-Type:application/json;charset=UTF-8");
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_POST, 1); //设置为非0表示本次操作为POST
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

  rapidjson::Document doc;
  doc.SetObject(); // key-value 相当与map

  rapidjson::Document::AllocatorType &allocator =
      doc.GetAllocator(); //获取分配器

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

  // end of for
  curl_easy_cleanup(curl);
  return true;
}

bool AlarmOutputModule::writeResult(AlgorithmResult const &rm,
                                    std::string &result) {
  rapidjson::Document doc;
  doc.SetObject(); // key-value 相当与map
  //获取分配器
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  if (!rm.bboxes.empty()) {
    rapidjson::Value bboxes(rapidjson::kArrayType);
    for (int i = 0; i < rm.bboxes.size(); i++) {
      // create a bbox object
      rapidjson::Value bbox;
      bbox.SetObject();
      // create coord array (x1, y1, x2, y2)
      rapidjson::Value coord(rapidjson::kArrayType);
      for (auto v : rm.bboxes[i].second) {
        coord.PushBack(v, allocator);
      }
      bbox.AddMember("coord", coord, allocator);
      rapidjson::Value className(rm.bboxes[i].first.c_str(), allocator);
      bbox.AddMember("class_name", className, allocator);
      bboxes.PushBack(bbox, allocator);
    }
    doc.AddMember("bboxes", bboxes, allocator);
  }

  if (!rm.polys.empty()) {
    rapidjson::Value polys(rapidjson::kArrayType);
    for (int i = 0; i < rm.polys.size(); i++) {
      // create a bbox object
      rapidjson::Value polygon;
      // create coord array (x1, y1, x2, y2)
      rapidjson::Value coord(rapidjson::kArrayType);
      for (auto v : rm.bboxes[i].second) {
        coord.PushBack(v, allocator);
      }
      polygon.AddMember("coord", coord, allocator);
      rapidjson::Value className(rm.polys[i].first.c_str(), allocator);
      polygon.AddMember("class_name", className, allocator);
      polys.PushBack(polygon, allocator);
    }
    doc.AddMember("bboxes", polys, allocator);
  }

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  result = std::string(buffer.GetString());
  return true;
}

void AlarmOutputModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      // FLOWENGINE_LOGGER_INFO("{} AlarmOutputModule module was done!", name);
      std::cout << name << "{} AlarmOutputModule module was done!" << std::endl;
      stopFlag.store(true);
      return;
    }
    if (recvModule.empty()) {
      return;
    }

    std::string algorithmInfo;
    writeResult(buf.algorithmResult, algorithmInfo);

    AlarmInfo alarmInfo{
        buf.cameraResult.heightPixel,
        buf.cameraResult.widthPixel,
        buf.cameraResult.cameraId,
        buf.alarmResult.eventId,
        buf.alarmResult.alarmVideoDuration,
        buf.alarmResult.page,
        buf.cameraResult.cameraIp,
        buf.alarmResult.alarmType,
        buf.alarmResult.alarmFile,
        buf.alarmResult.alarmId,
        buf.alarmResult.alarmDetails,
        algorithmInfo,
    };

    std::string response;
    if (!postResult(config.url, alarmInfo, response)) {
      FLOWENGINE_LOGGER_ERROR(
          "AlarmOutputModule.forward: post result was failed, please check!");
    }
  }
}

FlowEngineModuleRegister(AlarmOutputModule, Backend *, std::string const &,
                         std::string const &, common::OutputConfig const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &);
} // namespace module