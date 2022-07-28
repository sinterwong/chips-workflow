/**
 * @file sendOutputModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-06-05
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "sendOutputModule.h"
#include <opencv2/imgcodecs.hpp>

namespace module {
size_t curl_callback(void *ptr, size_t size, size_t nmemb, std::string *data) {
  data->append((char *)ptr, size * nmemb);
  return size * nmemb;
}

SendOutputModule::SendOutputModule(Backend *ptr,
                                   const std::string &initName,
                                   const std::string &initType, 
                                   const common::SendConfig &sendConfig,
                                   const std::vector<std::string> &recv,
                                   const std::vector<std::string> &send,
                                   const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool){
      url = sendConfig.url;
    }

bool SendOutputModule::postResult(std::string const &url,
                                  common::AlarmInfo const &alarmInfo,
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
  rapidjson::Value camera_id(alarmInfo.cameraId.c_str(), allocator);
  rapidjson::Value alarm_type(alarmInfo.alarmType.c_str(), allocator);
  rapidjson::Value alarm_file(alarmInfo.alarmFile.c_str(), allocator);
  rapidjson::Value alarm_id(alarmInfo.alarmId.c_str(), allocator);
  rapidjson::Value alarm_detail(alarmInfo.alarmDetails.c_str(), allocator);
  rapidjson::Value camera_ip(alarmInfo.cameraIp.c_str(), allocator);
  rapidjson::Value result_info(alarmInfo.resultInfo.c_str(), allocator);
  doc.AddMember("alarm_file", alarm_file, allocator);
  doc.AddMember("alarm_type", alarm_type, allocator);
  doc.AddMember("alarm_id", alarm_id, allocator);
  doc.AddMember("alarm_detail", alarm_detail, allocator);
  doc.AddMember("camera_ip", camera_ip, allocator);
  doc.AddMember("result_info", result_info, allocator);
  doc.AddMember("camera_id", camera_id, allocator);
  // --------------------
  doc.AddMember("host_id", alarmInfo.hostId, allocator);
  doc.AddMember("province_id", alarmInfo.provinceId, allocator);
  doc.AddMember("city_id", alarmInfo.cityId, allocator);
  doc.AddMember("region_id", alarmInfo.regionId, allocator);
  doc.AddMember("station_id", alarmInfo.stationId, allocator);
  doc.AddMember("width", alarmInfo.width, allocator);
  doc.AddMember("height", alarmInfo.height, allocator);
  doc.AddMember("location", alarmInfo.location, allocator);
  // */

  rapidjson::StringBuffer buffer;
  // PrettyWriter是格式化的json，如果是Writer则是换行空格压缩后的json
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);

  std::string s_out3 = std::string(buffer.GetString());

  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, s_out3.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, s_out3.length());

  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

  CURLcode response = curl_easy_perform(curl);

  // end of for
  curl_easy_cleanup(curl);
  return true;
}

bool SendOutputModule::writeResult(ResultMessage const &rm,
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

bool SendOutputModule::drawResult(cv::Mat &image, ResultMessage const &rm) {

  for (auto &bbox : rm.bboxes) {
    cv::Rect rect(bbox.second[0], bbox.second[1],
                  bbox.second[2] - bbox.second[0],
                  bbox.second[3] - bbox.second[1]);
    cv::rectangle(image, rect, cv::Scalar(255, 255, 0), 2);
    cv::putText(image, bbox.first, cv::Point(rect.x, rect.y - 1),
                cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 0, 255), 2);
  }

  for (auto &poly : rm.polys) {
    std::vector<cv::Point> fillContSingle;
    for (int i = 0; i < poly.second.size(); i += 2) {
      fillContSingle.emplace_back(
          cv::Point{static_cast<int>(poly.second[i]),
                    static_cast<int>(poly.second[i + 1])});
    }
    cv::fillPoly(image, std::vector<std::vector<cv::Point>>{fillContSingle},
                 cv::Scalar(0, 255, 255));
  }

  return true;
}

void SendOutputModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      FLOWENGINE_LOGGER_INFO("{} SendOutputModule module was done!", name);
      stopFlag.store(true);
    }
    if (recvModule.empty()) {
      return;
    }
    FrameBuf frameBufMessage = backendPtr->pool->read(buf.key);
    auto frame =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
    if (frame->empty()) {
      return;
    }

    // TODO 临时画个图
    cv::Mat showImage = frame->clone();
    if (buf.frameType == "RGB888") {
      cv::cvtColor(showImage, showImage, cv::COLOR_RGB2BGR);
    }
    drawResult(showImage, buf.results);
    cv::imwrite("/home/wangxt/workspace/projects/flowengine/tests/data/output.jpg", showImage);

    // resultTemplate.alarmFile = imageConverter.mat2str(showImage);

    // // resultTemplate.alarmFile = imageConverter.mat2str(*frame);
    // resultTemplate.alarmId = generate_hex(16);

    // writeResult(buf.results, resultTemplate.resultInfo);

    // std::string response;
    // if (!postResult(url, resultTemplate, response)) {
    //   FLOWENGINE_LOGGER_ERROR(
    //       "SendOutputModule.forward: post result was failed, please check!");
    // }
  }
}
} // namespace module