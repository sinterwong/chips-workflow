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
#include "core/messageBus.h"
#include "logger/logger.hpp"
#include <algorithm>
#include <array>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <string>
#include <type_traits>
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

void nms(DetRet &res, std::unordered_map<int, DetRet> &m, float nms_thr) {
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

void renderOriginShape(DetRet &results, std::array<int, 3> const &shape,
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

void YV12toNV12(cv::Mat const &input, cv::Mat &output) {
  int width = input.cols;
  int height = input.rows * 2 / 3;
  // Rows bytes stride - in most cases equal to width
  int stride = (int)input.step[0];
  cv::Mat temp = input.clone();

  // Y Channel
  //  YYYYYYYYYYYYYYYY
  //  YYYYYYYYYYYYYYYY
  //  YYYYYYYYYYYYYYYY
  //  YYYYYYYYYYYYYYYY
  //  YYYYYYYYYYYYYYYY
  //  YYYYYYYYYYYYYYYY

  // V Input channel
  //  VVVVVVVV
  //  VVVVVVVV
  //  VVVVVVVV
  cv::Mat inV =
      cv::Mat(cv::Size(width / 2, height / 2), CV_8UC1,
              (unsigned char *)temp.data + stride * height,
              stride / 2); // Input V color channel (in YV12 V is above U).

  // U Input channel
  //  UUUUUUUU
  //  UUUUUUUU
  //  UUUUUUUU
  cv::Mat inU =
      cv::Mat(cv::Size(width / 2, height / 2), CV_8UC1,
              (unsigned char *)temp.data + stride * height +
                  (stride / 2) * (height / 2),
              stride / 2); // Input V color channel (in YV12 U is below V).

  for (int row = 0; row < height / 2; row++) {
    for (int col = 0; col < width / 2; col++) {
      output.at<uchar>(height + row, 2 * col) = inU.at<uchar>(row, col);
      output.at<uchar>(height + row, 2 * col + 1) = inV.at<uchar>(row, col);
    }
  }
}

// template<typename common::ColorType Src=common::ColorType::RGB888>
void RGB2NV12(cv::Mat const &input, cv::Mat &output) {
  // 这里图片的宽高必须是偶数，否则直接卡死这里
  cv::cvtColor(input, output, cv::COLOR_RGB2YUV_YV12);
  YV12toNV12(output, output);
}

void NV12toRGB(cv::Mat const &nv12, cv::Mat &output) {
  cv::cvtColor(nv12, output, CV_YUV2RGB_NV12);
}

bool crop(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect, float sr) {
  if (sr > 0) {
    int sw = rect.width * sr;
    int sh = rect.height * sr;
    rect.x = std::max(0, rect.x - sw / 2);
    rect.y = std::max(0, rect.y - sh / 2);
    rect.width = std::min(input.cols - rect.x, rect.width + sw);
    rect.height = std::min(input.rows - rect.y, rect.height + sh);
  }
  // width 和 height 如果为奇数的话至少是1，因此可以直接操作
  if (rect.width % 2 != 0)
    rect.width -= 1;
  if (rect.height % 2 != 0) {
    rect.height -= 1;
  }
  output = input(rect).clone();
  return true;
}

bool cropImage(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect,
               common::ColorType type, float sr) {
  if (rect.width + rect.x > input.cols || rect.height + rect.y > input.rows) {
    FLOWENGINE_LOGGER_ERROR("cropImage is failed: error region!");
    return false;
  }
  switch (type) {
  case common::ColorType::RGB888:
  case common::ColorType::BGR888: {
    crop(input, output, rect, sr);
    break;
  }
  case common::ColorType::NV12: {
    // TODO 等实现了nv12专门的crop后替换此处的转换，目前的开销是不可接受的
    NV12toRGB(input, output);
    crop(output, output, rect, sr);
    RGB2NV12(output, output);
    break;
  }
  }
  return true;
}

} // namespace utils
} // namespace infer