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
#include <unordered_map>
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

// point 数据
using Point2i = std::array<int, 2>;
using Point2f = std::array<float, 2>;

// 单次分类结果
using ClsRet = std::pair<int, float>;

// 目标检测框集
using BBoxes = std::vector<BBox>;

// 字符识别结果
using CharsRet = std::vector<int>;

// 点集
using Points2i = std::vector<Point2i>;
using Points2f = std::vector<Point2f>;

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

// 特征
using Eigenvector = std::vector<float>;

// 算法结果
using AlgoRet = std::variant<std::monostate, BBoxes, ClsRet, CharsRet, Points2f,
                             KeypointsBoxes, Eigenvector>;

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
  float cond_thr;        // 置信度阈值
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
  int numPoints; // 特征维度
  float nmsThr; // NMS 阈值

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
  // 将所有参数类型存储在一个 std::variant 中
  using Params = std::variant<DetAlgo, ClassAlgo, FeatureAlgo>;

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
  LPRDet,
  CRNN,
  Softmax,
  FaceNet,
};

// 算法系列映射
static std::unordered_map<std::string, AlgoSerial> algoSerialMapping{
    std::make_pair("Yolo", AlgoSerial::Yolo),
    std::make_pair("Assd", AlgoSerial::Assd),
    std::make_pair("LPRDet", AlgoSerial::LPRDet),
    std::make_pair("CRNN", AlgoSerial::CRNN),
    std::make_pair("Softmax", AlgoSerial::Softmax),
    std::make_pair("FaceNet", AlgoSerial::FaceNet),
};

// 算法系列映射算法类型
static std::unordered_map<AlgoSerial, AlgoRetType> serial2TypeMapping{
    std::make_pair(AlgoSerial::Yolo, AlgoRetType::Detection),
    std::make_pair(AlgoSerial::Assd, AlgoRetType::Detection),
    std::make_pair(AlgoSerial::LPRDet, AlgoRetType::Detection),
    std::make_pair(AlgoSerial::CRNN, AlgoRetType::OCR),
    std::make_pair(AlgoSerial::Softmax, AlgoRetType::Classifier),
    std::make_pair(AlgoSerial::FaceNet, AlgoRetType::Feature),
};

/**
 * @brief 目前已经支持的算法种类
 *
 */
enum class SupportedAlgo : uint16_t {
  CocoDet = 0,
  HandDet,
  HeadDet,
  FireDet,
  SmogDet,
  SmokeCallCls,
  HelmetCls,
  ExtinguisherCls,
  OiltubeCls,
  EarthlineCls,
};

// TODO 支持的算法功能映射
static std::unordered_map<std::string, SupportedAlgo> algoMapping{
    std::make_pair("handDet", SupportedAlgo::HandDet),
    std::make_pair("headDet", SupportedAlgo::HeadDet),
    std::make_pair("phoneCls", SupportedAlgo::SmokeCallCls),
    std::make_pair("helmetCls", SupportedAlgo::HelmetCls),
    std::make_pair("smokeCls", SupportedAlgo::SmokeCallCls),
    std::make_pair("extinguisherCls", SupportedAlgo::ExtinguisherCls),
};

} // namespace common
#endif
