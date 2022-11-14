/**
 * @file infer_utils.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFERENCE_UTILS_H_
#define __INFERENCE_UTILS_H_
#include "infer_common.hpp"
#include "opencv2/imgproc.hpp"
#include <array>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

namespace infer {
namespace utils {
inline bool compare(DetectionResult const &a, DetectionResult const &b) {
  return a.det_confidence > b.det_confidence;
}

float iou(std::array<float, 4> const &, std::array<float, 4> const &);

void nms(std::vector<DetectionResult> &,
         std::unordered_map<int, std::vector<DetectionResult>> &, float);

void renderOriginShape(std::vector<DetectionResult> &results,
                       std::array<int, 3> const &shape,
                       std::array<int, 3> const &inputShape, bool isScale);

template <typename T>
inline void chw_to_hwc(T *chw_data, T *hwc_data, int channel, int height, int width) {
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
inline void hwc_to_chw(T *chw_data, T *hwc_data, int channel, int height, int width) {
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

bool resizeInput(cv::Mat &image, bool isScale, std::array<int, 2> &dstShape);

void BGR2YUV(const cv::Mat bgrImg, cv::Mat &y, cv::Mat &u, cv::Mat &v);

void YUV2BGR(const cv::Mat y, const cv::Mat u, const cv::Mat v,
             cv::Mat &bgrImg);

void NV12toRGB(cv::Mat &nv12, cv::Mat &output);

void YV12toNV12(const cv::Mat &input, cv::Mat &output);

void RGB2NV12(cv::Mat const &input, cv::Mat &output);

bool cropImage(cv::Mat const &input, cv::Mat &output, cv::Rect const &rect,
               common::ColorType type);

// TODO split nv12 -> y u v

// TODO rect nv12 -> calling to split nv12 to y u v and rect y u v separately,
// merge the result to new Mat

} // namespace utils
} // namespace infer
#endif
