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
#include "logger/logger.hpp"
#include <experimental/filesystem>
#include <fstream>

#include <nlohmann/json.hpp>
#include <vector>

using json = nlohmann::json;

namespace module::utils {
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

constexpr static bool is_base64(unsigned char c) {
  return (isalnum(c) || (c == '+') || (c == '/'));
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

bool retPolys2json(std::vector<RetPoly> const &retPolygons,
                   std::string &result) {
  if (retPolygons.empty()) {
    result = "{}";
    return false;
  }
  json polys;
  for (auto const &poly : retPolygons) {
    json polygon;
    json coord = poly.second; // array type
    polygon["coord"] = coord;
    polygon["class_name"] = poly.first;
    polys.push_back(polygon);
  }
  result = polys.dump();
  return true;
}

bool retBoxes2json(std::vector<RetBox> const &retBoxes, std::string &result) {
  if (retBoxes.empty()) {
    result = "{}";
    return false;
  }
  json bboxes;
  for (auto const &bbox : retBoxes) {
    json b;
    json coord = {bbox.x,      bbox.y,          bbox.width,
                  bbox.height, bbox.confidence, bbox.idx};
    b["coord"] = coord;
    b["class_name"] = bbox.name;
    bboxes.push_back(b);
  }
  result = bboxes.dump();
  return true;
}

bool retOCR2json(std::vector<OCRRet> const &retOCRs, std::string &result) {
  if (retOCRs.empty()) {
    result = "{}";
    return false;
  }
  json ocrs;
  for (auto const &ocr : retOCRs) {
    json kb;
    auto &b = ocr.kbbox.bbox.bbox;
    std::array<float, 6> bbox = {b[0],
                                 b[1],
                                 b[2],
                                 b[3],
                                 ocr.kbbox.bbox.class_confidence,
                                 ocr.kbbox.bbox.class_id};

    auto &ps = ocr.kbbox.points;
    json jbbox = bbox;
    kb["bbox"] = jbbox;

    std::vector<float> points;
    for (auto &p : ps) {
      points.push_back(p.x);
      points.push_back(p.y);
    }
    json jpoints = points;
    kb["points"] = jpoints;

    json jids = ocr.charIds;
    kb["ids"] = jids;

    kb["chars"] = ocr.chars;
    ocrs.push_back(kb);
  }
  result = ocrs.dump();
  return true;
}

bool drawRetBox(cv::Mat &image, std::vector<RetBox> const &bboxes,
                cv::Scalar const &scalar) {
  for (auto &bbox : bboxes) {
    cv::Rect rect(bbox.x, bbox.y, bbox.width, bbox.height);
    if (rect.area() == 0) {
      return false;
    }
    cv::rectangle(image, rect, scalar, 2);
    cv::putText(image, bbox.name, cv::Point(rect.x, rect.y - 1),
                cv::FONT_HERSHEY_PLAIN, 2, {255, 255, 255});
  }

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

bool readFile(std::string const &filename, std::string &ret) {
  std::ifstream input_file(filename);
  if (!input_file.is_open()) {
    FLOWENGINE_LOGGER_ERROR("Failed to open file: '{}'", filename);
    return false;
  }
  ret = std::string((std::istreambuf_iterator<char>(input_file)),
                    std::istreambuf_iterator<char>());
  return true;
}

bool writeJson(std::string const &config, std::string const &outPath) {
  FLOWENGINE_LOGGER_INFO("Writing json....");
  json j = json::parse(config);
  std::ofstream out(outPath);
  if (!out.is_open()) {
    FLOWENGINE_LOGGER_ERROR("Failed to open file: '{}'", outPath);
    return false;
  }
  out << j.dump(4);
  return true;
}

} // namespace module::utils