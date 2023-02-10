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
#include <opencv2/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/types.hpp>
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

void nms(DetRet &res, std::unordered_map<int, DetRet> &temp, float nms_thr) {
  for (auto it = temp.begin(); it != temp.end(); it++) {
    // std::cout << it->second[0].class_id << " --- " << std::endl;
    auto &dets = it->second;
    std::sort(dets.begin(), dets.end(), compare);
    for (size_t i = 0; i < dets.size(); ++i) {
      auto &item = dets[i];
      res.push_back(item);
      for (size_t j = i + 1; j < dets.size(); ++j) {
        if (iou(item.bbox, dets[j].bbox) > nms_thr) {
          dets.erase(dets.begin() + j);
          --j;
        }
      }
    }
  }
}

void restoryBoxes(DetRet &results, std::array<int, 3> const &shape,
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

void restoryPoints(PoseRet &results, std::array<int, 3> const &shape,
                   std::array<int, 3> const &inputShape, bool isScale) {}

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

void YU122NV12(cv::Mat const &input, cv::Mat &output) {
  int y_rows = input.rows * 2 / 3;
  cv::Mat y = input.rowRange(0, y_rows).colRange(0, input.cols);
  cv::Mat uv = input.rowRange(y_rows, input.rows).colRange(0, input.cols);
  output.create(cv::Size(input.cols, input.rows), CV_8UC1);

  // 拷贝y通道
  y.copyTo(output.rowRange(0, y.rows).colRange(0, y.cols)); // 5ms左右

  // // 指出nv12的uv通道，准备填值
  cv::Mat uvInterleaved =
      output.rowRange(y_rows, input.rows).colRange(0, uv.cols);

  // int u_rows = uv.rows / 2;
  // int u_cols = uv.cols / 2;
  // for (int row = 0; row < u_rows; ++row) {
  //   for (int col = 0; col < u_cols; ++col) {
  //     // 一共四组，每组复制总共30ms，四组共耗时大约120ms
  //     uvInterleaved.at<uchar>(2 * row, col * 2) = uv.at<uchar>(row, col); //
  //     u uvInterleaved.at<uchar>(2 * row, col * 2 + 1) = uv.at<uchar>(row +
  //     u_rows, col); // v

  //     uvInterleaved.at<uchar>(2 * row + 1, col * 2) = uv.at<uchar>(row, col +
  //     u_cols); // u uvInterleaved.at<uchar>(2 * row + 1, col * 2 + 1) =
  //     uv.at<uchar>(row + u_rows, col + u_cols); // v
  //   }
  // }

  // 将UV分别重塑为两个矩阵
  cv::Mat u = uv.rowRange(0, uv.rows / 2).colRange(0, uv.cols / 2);
  cv::Mat v = uv.rowRange(uv.rows / 2, uv.rows).colRange(0, uv.cols / 2);

  // 将U和V矩阵分别复制到UV通道中，每个U/V对应一个2x2的位置
  for (int row = 0; row < uv.rows / 2; ++row) {
    uchar *uvInterleavedRow = uvInterleaved.ptr<uchar>(2 * row);
    uchar *uRow = u.ptr<uchar>(row);
    uchar *vRow = v.ptr<uchar>(row);

    for (int col = 0; col < uv.cols / 2; ++col) {
      uvInterleavedRow[2 * col] = uRow[col];
      uvInterleavedRow[2 * col + 1] = vRow[col];
    }

    uvInterleavedRow = uvInterleaved.ptr<uchar>(2 * row + 1);

    for (int col = 0; col < uv.cols / 2; ++col) {
      uvInterleavedRow[2 * col] = uRow[col];
      uvInterleavedRow[2 * col + 1] = vRow[col];
    }
  }
}

void YUV444toNV12(cv::Mat const &input, cv::Mat &output) {
  output.create(cv::Size(input.cols, input.rows * 3 / 2), CV_8UC1);
  int uv_rows = input.rows / 2;
  int uv_cols = input.cols / 2;

  cv::Mat yuv_channel[3];
  cv::split(input, yuv_channel);
  cv::Mat y = yuv_channel[0];
  cv::Mat u = yuv_channel[1];
  cv::Mat v = yuv_channel[2];

  // 将U和V分量重采样为与Y分量具有相同的大小
  cv::Mat u_resized, v_resized;
  cv::resize(u, u_resized, cv::Size(uv_cols, uv_rows), 0, 0, cv::INTER_LINEAR);
  cv::resize(v, v_resized, cv::Size(uv_cols, uv_rows), 0, 0, cv::INTER_LINEAR);

  // 将重采样后的U和V分量并入一个矩阵中
  cv::Mat uv_interleaved =
      output.rowRange(y.rows, input.rows * 3 / 2).colRange(0, input.cols);
  for (int row = 0; row < uv_rows; ++row) {
    for (int col = 0; col < uv_cols; ++col) {
      uv_interleaved.at<uchar>(row, col * 2) = u_resized.at<uchar>(row, col);
      uv_interleaved.at<uchar>(row, col * 2 + 1) =
          v_resized.at<uchar>(row, col);
    }
  }
  // 将Y和UV分量并入一个矩阵中
  y.copyTo(output.rowRange(0, y.rows).colRange(0, y.cols));
}

// template<typename common::ColorType Src=common::ColorType::RGB888>
void RGB2NV12(cv::Mat const &input, cv::Mat &output) {
  // 这里图片的宽高必须是偶数，否则直接卡死这里
  cv::Mat temp;
  cv::cvtColor(input, temp, cv::COLOR_RGB2YUV_I420);
  YU122NV12(temp, output);

  // cv::cvtColor(input, temp, cv::COLOR_RGB2YUV);
  // YUV444toNV12(temp, output);
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

bool crop_nv12(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect,
               float sr) {
  // Get the width, height and stride of the input image
  int width = input.cols;
  int height = input.rows;
  int stride = static_cast<int>(input.step[0]);

  // Check if the rect is within the bounds of the input image
  if (rect.x < 0 || rect.y < 0 || rect.x + rect.width > width ||
      rect.y + rect.height > height) {
    return false;
  }

  // Get pointers to the input Y and UV channels
  const uint8_t *pY = input.ptr<uint8_t>(0);
  const uint8_t *pUV = pY + height * stride;

  // Create a new image for the output
  output = cv::Mat(rect.height, rect.width, CV_8UC2);

  // Get pointers to the output Y and UV channels
  uint8_t *pOutY = output.ptr<uint8_t>(0);
  uint8_t *pOutUV = pOutY + rect.height * output.step[0];

  // Loop through each row of the image
  for (int row = 0; row < rect.height; row++) {
    // Copy the Y channel
    memcpy(pOutY, pY + (row + rect.y) * stride + rect.x, rect.width);
    pOutY += output.step[0];

    // If this is an even row, copy the UV channels
    if ((row + rect.y) % 2 == 0) {
      memcpy(pOutUV, pUV + (row + rect.y) * stride / 2 + rect.x, rect.width);
      pOutUV += output.step[0];
    }
  }

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
  case common::ColorType::None: {
    FLOWENGINE_LOGGER_ERROR("cropImage is failed: unknow the image ColorType!");
    break;
  }
  }
  return true;
}

} // namespace utils
} // namespace infer