/**
 * @file trt_inference.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <algorithm>
#include <array>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "x3_inference.hpp"

namespace infer {
namespace x3 {
template <typename T>
void chw_to_hwc(T *chw_data, T *hwc_data, int channel, int height, int width) {
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
void hwc_to_chw(T *chw_data, T *hwc_data, int channel, int height, int width) {
  int wc = width * channel;
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

bool X3Inference::initialize() {
  return true;
}

bool X3Inference::infer(void *inputs, Result &result) {
  return true;
}

} // namespace trt
} // namespace infer
