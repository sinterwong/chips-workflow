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
#include <climits>
#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#ifndef _FLOWENGINE_COMMON_INFER_HEADER_HPP_
#define _FLOWENGINE_COMMON_INFER_HEADER_HPP_

namespace common {

// point 数据
template <typename T> struct Point {
  T x;
  T y;
};

// 单次分类结果
using ClsRet = std::pair<int, float>;

// 字符识别结果
using CharsRet = std::vector<int>;

// 点集
using Points2i = std::vector<Point<int>>;
using Points2f = std::vector<Point<float>>;

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
  Shape shape; // 图片分辨率
  Shape inputShape; // 图片输入算法时的维度（eg:NV12的话维度为{w, h * 1.5, c};
  ColorType type;
  void **data;
};

/**
 * @brief bbox result type, x1, y1, x2, y2, conf, class_id
 *
 */
// using RetBox = std::pair<std::string, std::array<float, 6>>;

struct RetBox {
  std::string name;
  Points2i points;
  int x = 0, y = 0, width = 0, height = 0, idx = 0;
  float confidence = 0.0;
  bool isPoly = false;

  RetBox() = default;

  RetBox(std::string name_, Points2i points_ = {})
      : name(std::move(name_)), points(std::move(points_)) {
    if (points.size() > 2) {
      isPoly = true;
    }
    getRectBox(points);
  }

  RetBox(std::string name_, int x_, int y_, int width_, int height_,
         float conf_ = 0.0, int idx_ = 0)
      : name(std::move(name_)), x(x_), y(y_), width(width_), height(height_),
        idx(idx_), confidence(conf_) {}

private:
  void getRectBox(Points2i &area) {
    int minX = INT_MAX, minY = INT_MAX, maxX = INT_MIN, maxY = INT_MIN;
    for (const auto &point : area) {
      minX = std::min(minX, point.x);
      minY = std::min(minY, point.y);
      maxX = std::max(maxX, point.x);
      maxY = std::max(maxY, point.y);
    }
    x = minX;
    y = minY;
    width = maxX - minX;
    height = maxY - minY;
  }
};

/**
 * @brief poly result type x1, y1, x2, y2, x3, y3, x4, y4, class_id
 *
 */
using RetPoly = std::pair<std::string, std::array<float, 9>>;

/**
 * @brief 算法推理时候需要使用的特定配置
 *
 */
struct InferParams {
  std::string name;    // 调用逻辑的名称
  ColorType frameType; // frameType
  float cropScaling;   // 截图时候需要缩放的比例
  RetBox region;       // bboxes
  Shape shape;         // 图像的尺寸
  // std::vector<int> attentionClasses; // [0, 2, 10...] logic中操作
};

/**
 * @brief 算法类型
 *
 */
enum class AlgoRetType : uint16_t {
  Classifier = 0,
  Detection,
  OCR,
  Pose,
  Feature,
};

/**
 * @brief 目标检测框
 *
 */
struct alignas(float) BBox {
  // x y w h
  std::array<float, 4> bbox; // [x1, y1, x2, y2]
  float class_confidence;
  float class_id;
  float det_confidence;
};

// 目标检测框集
using BBoxes = std::vector<BBox>;

// 关键点框
struct KeypointsBox {
  BBox bbox;       // 目标检测框
  Points2f points; // 框中的关键点
};

// 关键点框集
using KeypointsBoxes = std::vector<KeypointsBox>;

// OCR识别结果
struct OCRRet {
  KeypointsBox kbbox; // 一个四点框一个两点框
  CharsRet charIds;   // 字符识别结果
  std::string chars;  // 映射后结果
};

// 关键点检测（无框）
struct KeypointsRet {
  Points2f points; // 关键点
  float *M;        // 仿射变换矩阵
};

// 特征
using Eigenvector = std::vector<float>;

// 算法结果
using AlgoRet = std::variant<std::monostate, BBoxes, ClsRet, CharsRet,
                             KeypointsRet, KeypointsBoxes, Eigenvector>;

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
  int batchSize;                        // batch of number
  std::vector<std::string> inputNames;  // input names
  std::vector<std::string> outputNames; // output names
  std::string modelPath;                // 算法模型
  std::string serial; // 算法系列 {Yolo, Assd, Softmax, ...}
  Shape inputShape;   // 输入图像尺寸
  bool isScale;       // 预处理时是否等比例缩放
  float alpha;        // 预处理时除数
  float beta;         // 预处理时减数
  float cond_thr;     // 置信度阈值
};

/**
 * @brief 检测算法 (无所谓anchor-base和anchor-free, 后处理后都需要nms)
 *
 */
struct DetAlgo : public AlgoBase {
  float nmsThr; // NMS 阈值

  DetAlgo() = default;
  DetAlgo(AlgoBase &&algoBase, float nmsThr_)
      : AlgoBase(algoBase), nmsThr(nmsThr_) {}
};

/**
 * @brief 分类算法
 *
 */
struct ClassAlgo : public AlgoBase {
  ClassAlgo() = default;
  ClassAlgo(AlgoBase &&algoBase) : AlgoBase(algoBase) {}
};

/**
 * @brief 带有关键点的检测算法
 *
 */
struct PointsDetAlgo : public AlgoBase {
  int numPoints; // 点数
  float nmsThr;  // NMS 阈值

  PointsDetAlgo() = default;
  PointsDetAlgo(AlgoBase &&algoBase, int numPoints_, float nmsThr_)
      : AlgoBase(algoBase), numPoints(numPoints_), nmsThr(nmsThr_) {}
};

/**
 * @brief 特征提取算法，如人脸识别，行人ReID
 *
 */
struct FeatureAlgo : public AlgoBase {
  int dim; // 特征维度

  FeatureAlgo() = default;
  FeatureAlgo(AlgoBase &&algoBase, int dim_) : AlgoBase(algoBase), dim(dim_) {}
};

class AlgoConfig {
public:
  // 参数中心
  using Params =
      std::variant<AlgoBase, DetAlgo, ClassAlgo, FeatureAlgo, PointsDetAlgo>;

  // 设置参数
  template <typename T> void setParams(T params) {
    params_ = std::move(params);
  }

  // 访问参数
  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &params) { std::forward<Func>(func)(params); },
               params_);
  }

  // 获取参数
  template <typename T> T *getParams() { return std::get_if<T>(&params_); }

  // 获取copy参数
  template <typename T> T getCopyParams() { return std::get<T>(params_); }

  AlgoBase *getBaseParams() {
    return std::visit(
        [](auto &&arg) -> AlgoBase * {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_base_of_v<AlgoBase, T>) {
            return &arg;
          } else {
            return nullptr;
          }
        },
        params_);
  }

private:
  Params params_;
};

/**
 * @brief 支持的算法系列
 *
 */
enum class AlgoSerial : uint16_t {
  Yolo = 0,
  Assd,
  YoloPDet,
  Yolov8PDet,
  CRNN,
  Softmax,
  FaceNet,
  FaceKeyPoints,
};

// 算法系列映射
static std::unordered_map<std::string, AlgoSerial> algoSerialMapping{
    std::make_pair("Yolo", AlgoSerial::Yolo),
    std::make_pair("Assd", AlgoSerial::Assd),
    std::make_pair("YoloPDet", AlgoSerial::YoloPDet),
    std::make_pair("Yolov8PDet", AlgoSerial::Yolov8PDet),
    std::make_pair("CRNN", AlgoSerial::CRNN),
    std::make_pair("Softmax", AlgoSerial::Softmax),
    std::make_pair("FaceNet", AlgoSerial::FaceNet),
    std::make_pair("FaceKeyPoints", AlgoSerial::FaceKeyPoints),
};

// 算法系列映射算法类型
static std::unordered_map<AlgoSerial, AlgoRetType> serial2TypeMapping{
    std::make_pair(AlgoSerial::Yolo, AlgoRetType::Detection),
    std::make_pair(AlgoSerial::Assd, AlgoRetType::Detection),
    std::make_pair(AlgoSerial::YoloPDet, AlgoRetType::Detection),
    std::make_pair(AlgoSerial::Yolov8PDet, AlgoRetType::Detection),
    std::make_pair(AlgoSerial::FaceKeyPoints, AlgoRetType::Detection),
    std::make_pair(AlgoSerial::CRNN, AlgoRetType::OCR),
    std::make_pair(AlgoSerial::Softmax, AlgoRetType::Classifier),
    std::make_pair(AlgoSerial::FaceNet, AlgoRetType::Feature),
};
} // namespace common
#endif
