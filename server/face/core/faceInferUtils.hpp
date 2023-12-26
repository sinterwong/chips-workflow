/**
 * @file faceInferUtils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-05
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __SERVER_FACE_CORE_FACE_INFER_UTILS_HPP_
#define __SERVER_FACE_CORE_FACE_INFER_UTILS_HPP_

#include "common/infer_header.hpp"
#include "preprocess.hpp"
#include <opencv2/imgproc.hpp>


namespace server::face::core {
using common::getPlatformColorType;

inline void convertBGRToInputByType(cv::Mat const &inputBGR, cv::Mat &output) {
  auto type = getPlatformColorType();
  switch (type) {
  case common::ColorType::NV12: {
    infer::utils::BGR2NV12(inputBGR, output);
    break;
  }
  case common::ColorType::RGB888: {
    cv::cvtColor(inputBGR, output, cv::COLOR_BGR2RGB);
    break;
  }
  default:
    output = inputBGR;
    break;
  }
}

inline void convertInputToBGRByType(cv::Mat const &input, cv::Mat &outputBGR) {
  auto type = getPlatformColorType();
  switch (type) {
  case common::ColorType::NV12: {
    cv::cvtColor(input, outputBGR, cv::COLOR_YUV2BGR_NV12);
    break;
  }
  case common::ColorType::RGB888: {
    cv::cvtColor(input, outputBGR, cv::COLOR_RGB2BGR);
    break;
  }
  default:
    outputBGR = input;
    break;
  }
}
} // namespace server::face::core

#endif // __SERVER_FACE_CORE_FACE_INFER_UTILS_HPP_
