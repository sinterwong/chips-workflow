/**
 * @file preprocess.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-14
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __INFERENCE_UTILS_PREPROCESS_H_
#define __INFERENCE_UTILS_PREPROCESS_H_
#include "infer_common.hpp"
#include "logger/logger.hpp"
#include "opencv2/imgproc.hpp"
#include <utility>

namespace infer::utils {
template <typename T>
inline void chw_to_hwc(T *chw_data, T *hwc_data, int channel, int height,
                       int width) {
  int wc = width * channel;
  int index = 0;
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        hwc_data[h * wc + w * channel + c] = chw_data[index];
        index++;
      }
    }
  }
}

template <typename T>
inline void hwc_to_chw(T *chw_data, T *hwc_data, int channel, int height,
                       int width) {
  int wh = width * height;
  int index = 0;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int c = 0; c < channel; c++) {
        chw_data[c * wh + h * width + w] = hwc_data[index];
        index++;
      }
    }
  }
}

inline void normalize_L2(float *x, int d) {
  float sum = 0;
  for (int i = 0; i < d; i++) {
    sum += x[i] * x[i];
  }
  sum = std::sqrt(sum);
  for (int i = 0; i < d; i++) {
    x[i] /= sum;
  }
}

inline float dot(std::vector<float> &f1, std::vector<float> &f2) {
  // 归一化后的特征向量，计算点积
  float ret = 0.0;
  for (size_t i = 0; i < f1.size(); ++i) {
    ret += f1.at(i) * f2.at(i);
  }
  return ret;
}

bool resizeInput(cv::Mat &image, bool isScale, std::array<int, 2> &dstShape);

void BGR2YUV(const cv::Mat bgrImg, cv::Mat &y, cv::Mat &u, cv::Mat &v);

void YUV2BGR(const cv::Mat y, const cv::Mat u, const cv::Mat v,
             cv::Mat &bgrImg);

void NV12toRGB(cv::Mat const &nv12, cv::Mat &output);

void YV12toNV12(const cv::Mat &input, cv::Mat &output);

void RGB2NV12(cv::Mat const &input, cv::Mat &output, bool is_parallel = false);

void BGR2NV12(cv::Mat const &input, cv::Mat &output, bool is_parallel = false);

bool cropImage(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect,
               common::ColorType const &type, float sr = 0.0);

using common::RetBox;
bool cropImage(cv::Mat const &input, cv::Mat &output, RetBox &bbox,
               common::ColorType const &type, float sr = 0);

std::pair<float, float> sharpnessAndBrightnessScore(cv::Mat const &input);

inline common::ColorType getColorType() {
  if (TARGET_PLATFORM == 0 || TARGET_PLATFORM == 2) {
    return common::ColorType::NV12;
  } else if (TARGET_PLATFORM == 1) {
    return common::ColorType::RGB888;
  } else {
    return common::ColorType::BGR888;
  }
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
    BGR2NV12(inputBGR, output);
    break;
  }
  case common::ColorType::RGB888: {
    cv::cvtColor(inputBGR, output, cv::COLOR_BGR2RGB);
    break;
  }
  case common::ColorType::BGR888: {
    output = inputBGR;
    break;
  }
  default:
    FLOWENGINE_LOGGER_ERROR("Unsupported color type: {}",
                            static_cast<int>(type));
    throw std::runtime_error("Unsupported color type");
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
  case common::ColorType::BGR888: {
    outputBGR = input;
    break;
  }
  default:
    FLOWENGINE_LOGGER_ERROR("Unsupported color type: {}",
                            static_cast<int>(type));
    throw std::runtime_error("Unsupported color type");
    break;
  }
}

} // namespace infer::utils
#endif