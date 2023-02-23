/**
 * @file preprocess.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-14
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "preprocess.hpp"
#include "logger/logger.hpp"
#include <common/common.hpp>
#include <opencv2/imgproc/types_c.h>

using common::ColorType;

namespace infer {
namespace utils {

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
  //     uvInterleaved.at<uchar>(2 * row, col * 2) = uv.at<uchar>(row, col);
  //     uvInterleaved.at<uchar>(2 * row, col * 2 + 1) =
  //         uv.at<uchar>(row + u_rows, col);
  //     uvInterleaved.at<uchar>(2 * row + 1, col * 2) =
  //         uv.at<uchar>(row, col + u_cols);
  //     uvInterleaved.at<uchar>(2 * row + 1, col * 2 + 1) =
  //         uv.at<uchar>(row + u_rows, col + u_cols);
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

// template<typename ColorType Src=ColorType::RGB888>
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

template <ColorType format>
bool _crop(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect) {
  return false;
}

template <>
bool _crop<ColorType::RGB888>(cv::Mat const &input, cv::Mat &output,
                              cv::Rect2i &rect) {
  output = input(rect).clone();
  return true;
}

template <>
bool _crop<ColorType::NV12>(cv::Mat const &input, cv::Mat &output,
                            cv::Rect2i &rect) {
  cv::Mat y = input.rowRange(rect.y, rect.y + rect.height)
                  .colRange(rect.x, rect.x + rect.width);
  cv::Mat uv = input
                   .rowRange(input.rows * 2 / 3 + rect.y / 2,
                             input.rows * 2 / 3 + (rect.y + rect.height) / 2)
                   .colRange(rect.x, rect.x + rect.width);
  cv::Mat y_cropped = y.clone();
  cv::Mat uv_cropped = uv.clone();
  cv::Rect2i rect_y(0, 0, rect.width, rect.height);
  cv::Rect2i rect_uv(0, 0, rect.width, rect.height / 2);
  y_cropped = y_cropped(rect_y);
  uv_cropped = uv_cropped(rect_uv);
  cv::vconcat(y_cropped, uv_cropped, output);

  return true;
}

bool crop(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect,
          ColorType const type) {
  switch (type) {
  case ColorType::RGB888:
  case ColorType::BGR888: {
    return _crop<ColorType::RGB888>(input, output, rect);
  }
  case ColorType::NV12: {
    return _crop<ColorType::NV12>(input, output, rect);
  }
  case ColorType::None: {
    FLOWENGINE_LOGGER_ERROR("cropImage is failed: unknow the image ColorType!");
    return false;
  }
  }
  return false;
}

bool cropImage(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect,
               ColorType type, float sr) {
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
  if (rect.width + rect.x > input.cols || rect.height + rect.y > input.rows) {
    FLOWENGINE_LOGGER_ERROR("cropImage is failed: error region!");
    return false;
  }
  crop(input, output, rect, type);
  return true;
}

} // namespace utils
} // namespace infer