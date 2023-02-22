/**
 * @file module_utils.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "module_utils.hpp"

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include <vector>

namespace module {
namespace utils {
std::string base64_encode(uchar const *bytes_to_encode, unsigned int in_len) {
  std::string ret;

  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] =
          ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] =
          ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for (i = 0; (i < 4); i++) {
        ret += base64_chars[char_array_4[i]];
      }
      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 3; j++) {
      char_array_3[j] = '\0';
    }

    char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] =
        ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] =
        ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
    char_array_4[3] = char_array_3[2] & 0x3f;

    for (j = 0; (j < i + 1); j++) {
      ret += base64_chars[char_array_4[j]];
    }

    while ((i++ < 3)) {
      ret += '=';
    }
  }

  return ret;
}

std::string base64_decode(std::string const &encoded_string) {
  int in_len = encoded_string.size();
  int i = 0;
  int j = 0;
  int in_ = 0;
  unsigned char char_array_4[4], char_array_3[3];
  std::string ret;

  while (in_len-- && (encoded_string[in_] != '=') &&
         is_base64(encoded_string[in_])) {
    char_array_4[i++] = encoded_string[in_];
    in_++;

    if (i == 4) {
      for (i = 0; i < 4; i++) {
        char_array_4[i] = base64_chars.find(char_array_4[i]);
      }

      char_array_3[0] =
          (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
      char_array_3[1] =
          ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
      char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

      for (i = 0; (i < 3); i++) {
        ret += char_array_3[i];
      }

      i = 0;
    }
  }

  if (i) {
    for (j = i; j < 4; j++) {
      char_array_4[j] = 0;
    }

    for (j = 0; j < 4; j++) {
      char_array_4[j] = base64_chars.find(char_array_4[j]);
    }

    char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
    char_array_3[1] =
        ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
    char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

    for (j = 0; (j < i - 1); j++) {
      ret += char_array_3[j];
    }
  }

  return ret;
}

std::string mat2str(const cv::Mat &m) {
  int params[3] = {0};
  params[0] = cv::IMWRITE_JPEG_QUALITY;
  params[1] = 100;

  std::vector<uchar> buf;
  cv::imencode(".jpg", m, buf, std::vector<int>(params, params + 2));
  uchar *result = reinterpret_cast<uchar *>(&buf[0]);

  return base64_encode(result, buf.size());
}

cv::Mat str2mat(const std::string &s) {
  // Decode data
  std::string decoded_string = base64_decode(s);
  std::vector<uchar> data(decoded_string.begin(), decoded_string.end());

  cv::Mat img = imdecode(data, cv::IMREAD_UNCHANGED);
  return img;
}

bool retPolys2json(std::vector<RetPoly> const &retPolygons, std::string &result) {
  rapidjson::Document doc;
  doc.SetObject();
  // 获取分配器
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  if (!retPolygons.empty()) {
    rapidjson::Value polys(rapidjson::kArrayType);
    for (int i = 0; i < static_cast<int>(retPolygons.size()); i++) {
      rapidjson::Value polygon;
      polygon.SetObject();
      rapidjson::Value coord(rapidjson::kArrayType);
      for (auto v : retPolygons[i].second) {
        coord.PushBack(v, allocator);
      }
      polygon.AddMember("coord", coord, allocator);
      rapidjson::Value className(retPolygons[i].first.c_str(), allocator);
      polygon.AddMember("class_name", className, allocator);
      polys.PushBack(polygon, allocator);
    }
    doc.AddMember("polygons", polys, allocator);
  }

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  result = std::string(buffer.GetString());
  return true;
}

bool retBoxes2json(std::vector<RetBox> const &retBoxes, std::string &result) {
  rapidjson::Document doc;
  doc.SetObject();
  // 获取分配器
  rapidjson::Document::AllocatorType &allocator = doc.GetAllocator();

  if (!retBoxes.empty()) {
    rapidjson::Value bboxes(rapidjson::kArrayType);
    for (int i = 0; i < static_cast<int>(retBoxes.size()); i++) {
      rapidjson::Value bbox;
      bbox.SetObject();
      rapidjson::Value coord(rapidjson::kArrayType);
      for (auto v : retBoxes[i].second) {
        coord.PushBack(v, allocator);
      }
      bbox.AddMember("coord", coord, allocator);
      rapidjson::Value className(retBoxes[i].first.c_str(), allocator);
      bbox.AddMember("class_name", className, allocator);
      bboxes.PushBack(bbox, allocator);
    }
    doc.AddMember("bboxes", bboxes, allocator);
  }

  rapidjson::StringBuffer buffer;
  rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
  doc.Accept(writer);
  result = std::string(buffer.GetString());
  return true;
}

bool drawRetBox(cv::Mat &image, RetBox const &bbox, cv::Scalar const &scalar) {
  cv::Rect rect(bbox.second[0], bbox.second[1], bbox.second[2] - bbox.second[0],
                bbox.second[3] - bbox.second[1]);
  cv::rectangle(image, rect, scalar, 2);
  cv::putText(image, bbox.first, cv::Point(rect.x, rect.y - 1),
              cv::FONT_HERSHEY_PLAIN, 2, {255, 255, 255});
  return true;
}

bool drawRetPoly(cv::Mat &image, RetPoly const &poly,
                 cv::Scalar const &scalar) {
  std::vector<cv::Point> fillContSingle;
  for (int i = 0; i < static_cast<int>(poly.second.size()); i += 2) {
    fillContSingle.emplace_back(
        cv::Point{static_cast<int>(poly.second[i]),
                  static_cast<int>(poly.second[i + 1])});
  }
  cv::fillPoly(image, std::vector<std::vector<cv::Point>>{fillContSingle},
               cv::Scalar(0, 255, 255));

  return true;
}

} // namespace utils
} // namespace module