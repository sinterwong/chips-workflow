/**
 * @file common.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-04-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <array>
#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#ifndef _FLOWENGINE_COMMON_COMMON_HPP_
#define _FLOWENGINE_COMMON_COMMON_HPP_

namespace common {
/**
 * @brief 颜色类型
 *
 */
enum class ColorType { None = 0, RGB888, BGR888, NV12 };

/**
 * @brief 帧信息
 *
 */
struct FrameInfo {
  std::array<int, 3> shape;
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
  std::string name;                  // 调用逻辑的名称
  ColorType frameType;               // frameType
  float cropScaling;                 // 截图时候需要缩放的比例
  std::vector<RetBox> regions;       // bboxes
  std::vector<int> attentionClasses; // [0, 2, 10...]
};

/**
 * @brief bbox 数据
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
using Point = std::array<int, 3>;

// 单次分类结果
using ClsRet = std::pair<int, float>;

// 单次检测的结果
using DetRet = std::vector<BBox>;

// 单次关键点检测的结果
using PoseRet = std::vector<Point>;

/**
 * @brief 算法类型
 *
 */
enum class AlgoType {
  Classifier = 0,
  Detecton,
  Segmentation,
  KeyPoint,
  Feature
};

using AlgoRet = std::variant<std::monostate, DetRet, ClsRet, PoseRet>;

/**
 * @brief 算法推理时返回结果
 *
 */
struct InferResult {
  std::array<int, 3> shape; // 输入图像的尺寸
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
 * @brief no copy type
 *
 */
struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};

} // namespace common
#endif
