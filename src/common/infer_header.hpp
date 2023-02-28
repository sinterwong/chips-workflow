/**
 * @file infer_header.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-28
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <array>
#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#ifndef _FLOWENGINE_COMMON_INFER_HEADER_HPP_
#define _FLOWENGINE_COMMON_INFER_HEADER_HPP_

namespace common {

using svector = std::vector<std::string>;

/**
 * @brief 颜色类型
 *
 */
enum class ColorType {
  None = 0, // Raw type
  RGB888,   // RGB type
  BGR888,   // BGR type
  NV12      // YUV420 NV12 type
};

using Shape = std::array<int, 3>;

/**
 * @brief 帧信息
 *
 */
struct FrameInfo {
  Shape shape;
  ColorType type;
  void **data;
};

/**
 * @brief bbox result type
 *
 */
using RetBox = std::pair<std::string, std::array<float, 6>>;
/**
 * @brief poly result type
 *
 */
using RetPoly = std::pair<std::string, std::array<float, 9>>;

/**
 * @brief 算法推理时候需要使用的特定配置
 *
 */
struct InferParams {
  std::string name;            // 调用逻辑的名称
  ColorType frameType;         // frameType
  float cropScaling;           // 截图时候需要缩放的比例
  std::vector<RetBox> regions; // bboxes
  Shape shape;                 // 图像的尺寸
  // std::vector<int> attentionClasses; // [0, 2, 10...] logic中操作
};

/**
 * @brief 目标检测框
 *
 */
struct alignas(float) BBox {
  // x y w h
  std::array<float, 4> bbox; // [x1, y1, x2, y2]
  float det_confidence;
  float class_id;
  float class_confidence;
};

// point 数据
using Point = std::array<int, 2>;

// 单次分类结果
using ClsRet = std::pair<int, float>;

// 目标检测框集
using BBoxes = std::vector<BBox>;

// 点集
using Points = std::vector<Point>;

// 算法结果
using AlgoRet = std::variant<std::monostate, BBoxes, ClsRet, Points>;

/**
 * @brief 算法推理时返回结果
 *
 */
struct InferResult {
  Shape shape; // 输入图像的尺寸
  AlgoRet aRet;
};

/**
 * @brief 模型的基础信息（模型装载后可以获取）
 *
 */
struct ModelInfo {
  int output_count;                           // 输出的个数
  std::vector<std::vector<int>> outputShapes; // 输出的尺度
};

/**
 * @brief CV算法的基本参数
 *
 */
struct AlgoBase {
  int batchSize;         // batch of number
  svector inputNames;    // input names
  svector outputNames;   // output names
  std::string modelPath; // 算法模型
  std::string serial;    // 算法系列 {Yolo, Assd, Softmax, ...}
  Shape inputShape;      // 输入图像尺寸
  bool isScale;          // 预处理时是否等比例缩放
  float alpha;           // 预处理时除数
  float beta;            // 预处理时减数
};

/**
 * @brief 检测算法 (无所谓anchor-base和anchor-free, 后处理后都需要nms)
 *
 */
struct DetAlgo : public AlgoBase {
  float cond_thr; // 置信度阈值
  float nms_thr;  // NMS 阈值
};

/**
 * @brief 分类算法
 *
 */
struct ClassAlgo : public AlgoBase {};

} // namespace common
#endif
