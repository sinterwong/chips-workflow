/**
 * @file logicModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-10
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_LOGIC_MODULE_H_
#define __METAENGINE_LOGIC_MODULE_H_

#include <any>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <vector>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include "utils/convertMat.hpp"
#include "videoOutput.h"

namespace module {
class LogicModule : public Module {
protected:
  bool isRecord = false; // 是否是保存视频状态
  int frameCount = 0;
  std::unique_ptr<videoOutput> outputStream;
  common::LogicConfig params;           // 逻辑参数
  utils::ImageConverter imageConverter; // mat to base64

  inline unsigned int random_char() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    return dis(gen);
  }

  inline std::string generate_hex(const unsigned int len) {
    std::stringstream ss;
    for (auto i = 0; i < len; i++) {
      const auto rc = random_char();
      std::stringstream hexstream;
      hexstream << std::hex << rc;
      auto hex = hexstream.str();
      ss << (hex.length() < 2 ? '0' + hex : hex);
    }
    return ss.str();
  }

public:
  LogicModule(Backend *ptr, const std::string &initName,
              const std::string &initType, const common::LogicConfig &params_,
              const std::vector<std::string> &recv = {},
              const std::vector<std::string> &send = {},
              const std::vector<std::string> &pool = {})
      : Module(ptr, initName, initType, recv, send, pool), params(params_) {}
  virtual ~LogicModule() {}

  bool drawResult(cv::Mat &image, AlgorithmResult const &rm) {
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
};
} // namespace module
#endif // __METAENGINE_LOGIC_MODULE_H_
