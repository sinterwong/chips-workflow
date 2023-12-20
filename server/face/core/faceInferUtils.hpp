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
inline common::ColorType getColorType() {
  if (TARGET_PLATFORM == 0 || TARGET_PLATFORM == 2) {
    return common::ColorType::NV12;
  }
  return common::ColorType::RGB888;
}

inline common::Shape getShape(int width, int height) {
  auto type = getColorType();
  switch (type) {
  case common::ColorType::NV12: {
    return {width, height * 2 / 3, 3};
  }
  default:
    return {width, height, 3};
  }
}

inline void convertBGRToInputByType(cv::Mat const &inputBGR, cv::Mat &output) {
  auto type = getColorType();
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
  auto type = getColorType();
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
