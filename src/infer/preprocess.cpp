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
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.inl.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

using common::ColorType;

namespace infer::utils {

void interleaveRows(const cv::Mat &input1, const cv::Mat &input2,
                    cv::Mat &output) {
  // 确保两个输入矩阵具有相同的维度和数据类型
  CV_Assert(input1.size() == input2.size() && input1.type() == input2.type());

  // 调整输出矩阵的尺寸
  if (output.empty()) {
    output.create(input1.rows * 2, input1.cols, input1.type());
  }
  CV_Assert(input1.rows * 2 == output.rows);

  for (int row = 0; row < input1.rows; ++row) {
    // 为input1和input2的当前行获取指针
    const uchar *ptr1 = input1.ptr<uchar>(row);
    const uchar *ptr2 = input2.ptr<uchar>(row);

    // 获取output的两个相应行的指针
    uchar *outPtr1 = output.ptr<uchar>(row * 2);
    uchar *outPtr2 = output.ptr<uchar>(row * 2 + 1);

    // 复制input1和input2的内容到output的相应位置
    memcpy(outPtr1, ptr1, input1.cols * sizeof(uchar));
    memcpy(outPtr2, ptr2, input2.cols * sizeof(uchar));
  }
}

void interleaveColumns(const cv::Mat &input1, const cv::Mat &input2,
                       cv::Mat &output) {
  // 确保两个输入矩阵具有相同的维度和数据类型
  CV_Assert(input1.size() == input2.size() && input1.type() == input2.type());

  // 调整输出矩阵的尺寸
  if (output.empty()) {
    output.create(input1.rows, input1.cols * 2, input1.type());
  }

  CV_Assert(input1.cols * 2 == output.cols);

  // 比较低效，使用at()的方式获取值
  // for (int y = 0; y < output.rows; y++) {
  //   for (int x = 0; x < output.cols; x += 2) {
  //     output.at<uchar>(y, x) = input1.at<uchar>(y, x / 2);
  //     output.at<uchar>(y, x + 1) = input2.at<uchar>(y, x / 2);
  //   }
  // }

  // 相对高效，使用指针来直接访问Mat数据
  for (int row = 0; row < input1.rows; ++row) {
    const uchar *ptr1 = input1.ptr<uchar>(row);
    const uchar *ptr2 = input2.ptr<uchar>(row);
    uchar *outPtr = output.ptr<uchar>(row);

    for (int col = 0; col < input1.cols; ++col) {
      *outPtr++ = *ptr1++;
      *outPtr++ = *ptr2++;
    }
  }
}

cv::Mat interleaveColumnsParallel(const cv::Mat &input1, const cv::Mat &input2,
                                  cv::Mat &output) {
  CV_Assert(input1.size() == input2.size() && input1.type() == input2.type());

  if (output.empty()) {
    output.create(input1.rows, input1.cols * 2, input1.type());
  }

  cv::parallel_for_(cv::Range(0, input1.rows), [&](const cv::Range &range) {
    for (int row = range.start; row < range.end; row++) {
      const uchar *ptr1 = input1.ptr<uchar>(row);
      const uchar *ptr2 = input2.ptr<uchar>(row);
      uchar *outPtr = output.ptr<uchar>(row);

      for (int col = 0; col < input1.cols; ++col) {
        *outPtr++ = *ptr1++;
        *outPtr++ = *ptr2++;
      }
    }
  });

  return output;
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

bool delBoundaryInfo(cv::Mat const &input, cv::Mat &output,
                     std::vector<cv::Point2i> &points, ColorType type) {

  if (type == ColorType::NV12) {

    cv::Mat yMask(cv::Size(input.cols, input.rows * 2 / 3), CV_8UC1,
                  cv::Scalar(0));

    cv::fillPoly(yMask, std::vector<std::vector<cv::Point2i>>{points},
                 cv::Scalar(255));

    // u和v分量的mask
    cv::Mat uMask, vMask;
    cv::resize(yMask, uMask, cv::Size(yMask.cols / 2, yMask.rows / 2));
    vMask = uMask;

    cv::Mat uvMask;
    // 合并uMask和vMask到uvMask(uv是交叉排列的)
    // interleaveColumnsParallel(uMask, vMask, uvMask);
    interleaveColumns(uMask, vMask, uvMask);

    // 将yMask为0的部分对应的output填0
    output(cv::Rect(0, 0, input.cols, input.rows * 2 / 3)).setTo(0, yMask == 0);

    // 将uvMask为0的部分对应的output填128
    output(cv::Rect(0, input.rows * 2 / 3, input.cols, input.rows / 3))
        .setTo(128, uvMask == 0);

  } else if (type == ColorType::BGR888 || type == ColorType::RGB888) {
    // 创建一个空的黑色掩膜
    cv::Mat mask(output.size(), CV_8UC1, cv::Scalar(0));
    cv::fillPoly(mask, std::vector<std::vector<cv::Point2i>>{points},
                 cv::Scalar(255));
    output.setTo(0, mask == 0);
  } else {
    FLOWENGINE_LOGGER_ERROR(
        "delBoundaryInfo failed: unknow the image ColorType!");
    return false;
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
  // uv 不可以为奇数
  cv::Mat uv = input.rowRange(y_rows, input.rows).colRange(0, input.cols);

  // 对uv进行矫正
  cv::Rect rect = {0, 0, uv.cols, uv.rows};
  if (rect.height % 2) {
    rect.height -= 1;
    uv = uv(rect);
    if ((uv.rows + y.rows) % 3 != 0) {
      y_rows -= (uv.rows + y.rows) % 3;
      rect = {0, 0, y.cols, y_rows};
      y = y(rect);
    }
  }
  cv::Mat u = uv.rowRange(0, uv.rows / 2).colRange(0, uv.cols / 2);
  cv::Mat v = uv.rowRange(uv.rows / 2, uv.rows).colRange(0, uv.cols / 2);
  // 单个的一轮uv
  cv::Mat uvOne;
  // 两轮uv的交错
  cv::Mat uvInterleave;
  interleaveColumns(u, v, uvOne);
  interleaveRows(uvOne, uvOne, uvInterleave);
  // 合并最终的结果
  cv::vconcat(y, uvInterleave, output);
}

void YU122NV12_parallel(cv::Mat const &input, cv::Mat &output) {
  int y_rows = input.rows * 2 / 3;
  cv::Mat y = input.rowRange(0, y_rows).colRange(0, input.cols);
  cv::Mat uv = input.rowRange(y_rows, input.rows).colRange(0, input.cols);
  output.create(cv::Size(input.cols, input.rows), CV_8UC1);

  y.copyTo(output.rowRange(0, y.rows).colRange(0, y.cols));

  cv::Mat uvInterleaved =
      output.rowRange(y_rows, input.rows).colRange(0, uv.cols);

  cv::Mat u = uv.rowRange(0, uv.rows / 2).colRange(0, uv.cols / 2);
  cv::Mat v = uv.rowRange(uv.rows / 2, uv.rows).colRange(0, uv.cols / 2);

  // Use cv::parallel_for_ to parallelize the filling of uvInterleaved
  cv::parallel_for_(cv::Range(0, uv.rows / 2), [&](const cv::Range &range) {
    for (int row = range.start; row < range.end; ++row) {
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
  });
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
void RGB2NV12(cv::Mat const &input, cv::Mat &output, bool is_parallel) {
  // 这里图片的宽高必须是偶数，否则直接卡死这里
  cv::Rect rect{0, 0, input.cols, input.rows};
  rect.width -= rect.width % 2;
  rect.height -= rect.height % 2;
  cv::Mat in_temp = input(rect);
  cv::Mat temp;
  cv::cvtColor(in_temp, temp, cv::COLOR_RGB2YUV_I420);
  if (is_parallel) {
    YU122NV12_parallel(temp, output);
  } else {
    YU122NV12(temp, output);
  }

  // cv::cvtColor(input, temp, cv::COLOR_RGB2YUV);
  // YUV444toNV12(temp, output);
}

void BGR2NV12(cv::Mat const &input, cv::Mat &output, bool is_parallel) {
  // 这里图片的宽高必须是偶数，否则直接卡死这里
  cv::Rect rect{0, 0, input.cols, input.rows};
  rect.width -= rect.width % 2;
  rect.height -= rect.height % 2;
  cv::Mat in_temp = input(rect);
  cv::Mat temp;
  cv::cvtColor(in_temp, temp, cv::COLOR_BGR2YUV_I420);
  if (is_parallel) {
    YU122NV12_parallel(temp, output);
  } else {
    YU122NV12(temp, output);
  }

  // cv::cvtColor(input, temp, cv::COLOR_RGB2YUV);
  // YUV444toNV12(temp, output);
}

void NV12toRGB(cv::Mat const &nv12, cv::Mat &output) {
  cv::Rect rect{0, 0, nv12.cols, nv12.rows};
  // width % 2 == 0 and height % 3 == 0
  rect.width -= rect.width % 2;
  rect.height -= rect.height % 3;
  cv::Mat temp = nv12(rect);
  cv::cvtColor(temp, output, cv::COLOR_YUV2RGB_NV12);
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
  cv::Rect2i rect_y(0, 0, rect.width, rect.height);
  cv::Rect2i rect_uv(0, 0, rect.width, rect.height / 2);
  cv::Mat y_cropped = y(rect_y);
  cv::Mat uv_cropped = uv(rect_uv);
  cv::vconcat(y_cropped, uv_cropped, output);
  return true;
}

bool crop(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect,
          ColorType const type, float sr) {
  // 图片像素
  int maxWidth = input.cols;
  int maxHeight = type != ColorType::NV12 ? input.rows : input.rows / 3.0 * 2;
  if (sr > 0) {
    int sw = rect.width * sr;
    int sh = rect.height * sr;
    rect.x = std::max(0, rect.x - sw / 2);
    rect.y = std::max(0, rect.y - sh / 2);
    rect.width = std::min(maxWidth - rect.x, rect.width + sw);
    rect.height = std::min(maxHeight - rect.y, rect.height + sh);
  }
  // width 和 height 如果为奇数的话至少是1，因此可以直接操作
  rect.width -= rect.width % 2;
  rect.height -= rect.height % 2;

  if (rect.width + rect.x > maxWidth || rect.height + rect.y > maxHeight) {
    FLOWENGINE_LOGGER_ERROR("cropImage is failed: error region!");
    return false;
  }
  switch (type) {
  case ColorType::RGB888:
  case ColorType::BGR888: {
    return _crop<ColorType::RGB888>(input, output, rect);
  }
  case ColorType::NV12: {
    return _crop<ColorType::NV12>(input, output, rect);
  }
  default: {
    FLOWENGINE_LOGGER_ERROR("cropImage is failed: unknow the image ColorType!");
    return false;
  }
  }
  return false;
}

bool cropImage(cv::Mat const &input, cv::Mat &output, cv::Rect2i &rect,
               ColorType const &type, float sr) {
  return crop(input, output, rect, type, sr);
}

bool cropImage(cv::Mat const &input, cv::Mat &output, RetBox &bbox,
               ColorType const &type, float sr) {
  // 先抠图，如果是多边区域的话，就将多变区域描黑
  cv::Rect2i rect{bbox.x, bbox.y, bbox.width, bbox.height};
  if (sr > 0) { // 如果需要放大边框，直接不用扣了
    return cropImage(input, output, rect, type, sr);
  }
  if (!crop(input, output, rect, type, sr)) {
    return false;
  };
  if (bbox.isPoly) {
    // 为points做坐标偏移
    std::vector<cv::Point2i> offsetPoints;
    for (const auto &pt : bbox.points) {
      offsetPoints.emplace_back(pt.x - bbox.x, pt.y - bbox.y);
    }
    // 去除边界信息
    delBoundaryInfo(output, output, offsetPoints, type);
  }
  return true;
}

std::pair<float, float> sharpnessAndBrightnessScore(cv::Mat const &image) {
  // Crop the face from the image
  cv::Mat imageGray;

  cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

  // Blur the face image with a 5x5 Gaussian kernel
  cv::Mat blurFaceImage;
  cv::GaussianBlur(imageGray, blurFaceImage, cv::Size(5, 5), 1, 1);

  // Calculate the sharpness score
  cv::Mat diffImage;
  cv::absdiff(imageGray, blurFaceImage, diffImage);

  double sum = cv::sum(diffImage)[0];

  double sharpnessScore = sum / (diffImage.rows * diffImage.cols * 255.0);
  sharpnessScore = std::min(1.0, sharpnessScore * 100);

  // Calculate the brightness score
  double brightnessScore = cv::mean(imageGray)[0];
  if (brightnessScore < 20 || brightnessScore > 230) {
    brightnessScore = 0;
  } else {
    brightnessScore = 1 - std::abs(brightnessScore - 127.5) / 127.5;
  }
  return {static_cast<float>(sharpnessScore),
          static_cast<float>(brightnessScore)};
}
} // namespace infer::utils