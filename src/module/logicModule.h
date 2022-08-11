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
#include <random>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include "utils/convertMat.hpp"
#include "videoOutput.h"


namespace module {
class LogicModule : public Module {
protected:
  bool isRecord = false;  // 是否是保存视频状态
  int frameCount = 0;
  std::unique_ptr<videoOutput> outputStream;
  common::LogicConfig params;  // 逻辑参数
  utils::ImageConverter imageConverter;  // mat to base64

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
              const std::string &initType,
              const common::LogicConfig &params_,
              const std::vector<std::string> &recv = {},
              const std::vector<std::string> &send = {},
              const std::vector<std::string> &pool = {});
  virtual ~LogicModule() {}

  bool drawResult(cv::Mat &image, AlgorithmResult const &rm);
};
} // namespace module
#endif // __METAENGINE_LOGIC_MODULE_H_
