/**
 * @file infer_utils.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "infer_utils.hpp"
#include <algorithm>
#include <array>
namespace infer {
namespace utils {
float iou(std::array<float, 4> const &lbox, std::array<float, 4> const &rbox) {
  float interBox[] = {
      std::max(lbox[0] - lbox[2] / 2.f, rbox[0] - rbox[2] / 2.f), // left
      std::min(lbox[0] + lbox[2] / 2.f, rbox[0] + rbox[2] / 2.f), // right
      std::max(lbox[1] - lbox[3] / 2.f, rbox[1] - rbox[3] / 2.f), // top
      std::min(lbox[1] + lbox[3] / 2.f, rbox[1] + rbox[3] / 2.f), // bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
  return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(std::vector<DetectionResult> &res,
         std::unordered_map<int, std::vector<DetectionResult>> &m,
         float nms_thr) {
  for (auto it = m.begin(); it != m.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    auto &dets = it->second;
    std::sort(dets.begin(), dets.end(), compare);
    for (size_t m = 0; m < dets.size(); ++m) {
      auto &item = dets[m];
      res.push_back(item);
      for (size_t n = m + 1; n < dets.size(); ++n) {
        if (iou(item.bbox, dets[n].bbox) > nms_thr) {
          dets.erase(dets.begin() + n);
          --n;
        }
      }
    }
  }
}

void renderOriginShape(std::vector<DetectionResult> &results,
                       std::array<int, 3> const &shape,
                       std::array<int, 3> const &inputShape, bool isScale) {
  float rw, rh;
  if (isScale) {
    rw = std::min(inputShape[0] * 1.0 / shape.at(0),
                  inputShape[1] * 1.0 / shape.at(1));
    rh = rw;
  } else {
    rw = inputShape[0] * 1.0 / shape.at(0);
    rh = inputShape[1] * 1.0 / shape.at(1);
  }

  for (auto &ret : results) {
    int l = (ret.bbox[0] - ret.bbox[2] / 2.f) / rw;
    int t = (ret.bbox[1] - ret.bbox[3] / 2.f) / rh;
    int r = (ret.bbox[0] + ret.bbox[2] / 2.f) / rw;
    int b = (ret.bbox[1] + ret.bbox[3] / 2.f) / rh;
    ret.bbox[0] = l > 0 ? l : 0;
    ret.bbox[1] = t > 0 ? t : 0;
    ret.bbox[2] = r < shape[0] ? r : shape[0];
    ret.bbox[3] = b < shape[1] ? b : shape[1];
  }
}

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

bool resizeInput(cv::Mat &image, bool isScale, std::array<int, 2> &dstShape) {
  if (isScale) {
    int height = image.rows;
    int width = image.cols;
    float ratio =
        std::min(dstShape[0] * 1.0 / width, dstShape[1] * 1.0 / height);

    int dw = width * ratio;
    int dh = height * ratio;
    cv::resize(image, image, cv::Size(dw, dh));
    cv::copyMakeBorder(image, image, 0, std::max(0, dstShape[1] - dh), 0,
                       std::max(0, dstShape[0] - dw), cv::BORDER_CONSTANT,
                       cv::Scalar(128, 128, 128));
  } else {
    cv::resize(image, image, cv::Size(dstShape[0], dstShape[1]));
  }
  return true;
}

// BGR 转 YUV
void BGR2YUV(const cv::Mat bgrImg, cv::Mat &y, cv::Mat &u, cv::Mat &v) {
  cv::Mat out;

  cv::cvtColor(bgrImg, out, cv::COLOR_BGR2YUV);
  cv::Mat channel[3];
  cv::split(out, channel);
  y = channel[0];
  u = channel[1];
  v = channel[2];
}

// YUV 转 BGR
void YUV2BGR(const cv::Mat y, const cv::Mat u, const cv::Mat v,
             cv::Mat &bgrImg) {
  std::vector<cv::Mat> inChannels;
  inChannels.push_back(y);
  inChannels.push_back(u);
  inChannels.push_back(v);

  // 合并3个单独的 channel 进一个矩阵
  cv::Mat yuvImg;
  cv::merge(inChannels, yuvImg);

  cv::cvtColor(yuvImg, bgrImg, cv::COLOR_YUV2BGR);
}

} // namespace utils
} // namespace infer