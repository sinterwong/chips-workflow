/**
 * @file config.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifndef _FLOWENGINE_COMMON_CONFIG_HPP_
#define _FLOWENGINE_COMMON_CONFIG_HPP_

namespace common {

enum class ConfigType { Algorithm, Stream, Output, Logic, None };
enum class ModuleType {
  Detection = 0,
  Classifier,
  WebStream,
  AlarmOutput,
  StatusOutput,
  Calling,
  Smokeing
};

struct ModuleConfigure {
  std::string typeName;
  std::string ctype;
  std::string moduleName; // 模块名称
  std::string sendName;   // 下游模块
  std::string recvName;   // 上游模块
};

struct AlgorithmConfig {
  AlgorithmConfig() = default;
  AlgorithmConfig(std::string const &modelPath_,
                  std::vector<std::string> const &inputTensorNames_,
                  std::vector<std::string> const &outputTensorNames_,
                  std::array<int, 3> const &inputShape_, float cond_thr_ = 0.3,
                  float nms_thr_ = 0.5, float alpha_ = 255.0, float beta_ = 0.0,
                  bool isScale_ = true, int batchSize_ = 1)
      : modelPath(modelPath_), inputTensorNames(std::move(inputTensorNames_)),
        outputTensorNames(std::move(outputTensorNames_)),
        inputShape(std::move(inputShape_)), cond_thr(cond_thr_),
        nms_thr(nms_thr_), alpha(alpha_), beta(beta_), isScale(isScale_),
        batchSize(batchSize_){};
  // 算法配置
  std::string modelPath;                      // engine 所在目录
  std::vector<std::string> inputTensorNames;  // input tensor names
  std::vector<std::string> outputTensorNames; // output tensor names
  std::array<int, 3> inputShape;              // 算法需要的输入尺度
  bool isScale;                               // 是否等比例缩放
  float cond_thr;                             // 置信度阈值
  float nms_thr;                              // NMS 阈值
  float alpha;                                // 预处理时除数
  float beta;                                 // 预处理时减数
  int batchSize;                              // batch of number
};

struct CameraConfig {
  CameraConfig(std::string const &name_, std::string const &videoCode_,
               std::string const &flowType_, std::string const &cameraIp_,
               int widthPixel_, int heightPixel_, int cameraId_)
      : cameraName(name_), videoCode(videoCode_), flowType(flowType_),
        cameraIp(cameraIp_), widthPixel(widthPixel_), heightPixel(heightPixel_),
        cameraId(cameraId_) {}
  CameraConfig() = default;
  ~CameraConfig() {}
  // 摄像机配置
  int widthPixel;         // 视频宽度
  int heightPixel;        // 视频高度
  int cameraId;           // 摄像机 ID
  std::string cameraName; // 摄像机名称(uuid)
  std::string videoCode;  // 视频编码类型
  std::string flowType;   // 流协议类型
  std::string cameraIp;   // 网络流链接
};

struct LogicConfig {
  LogicConfig() = default;
  LogicConfig(std::string const &outputDir_, std::array<int, 4> const &region_,
              std::vector<int> const &attentionClasses_, int eventId_,
              std::string page_, int videoDuration_ = 0, float threshold_ = 0.9)
      : outputDir(outputDir_), region(region_),
        attentionClasses(attentionClasses_), eventId(eventId_), page(page_),
        videDuration(videoDuration_), threshold(threshold_) {}
  ~LogicConfig() {}
  std::string outputDir;
  int eventId;
  int videDuration;
  std::string page;
  std::array<int, 4> region;         // 划定区域
  std::vector<int> attentionClasses; // 划定区域
  float threshold;
  bool isDraw = true;
};

struct OutputConfig {
  OutputConfig() = default;
  OutputConfig(std::string const &url_) : url(url_) {}
  ~OutputConfig() {}
  std::string url;
};

class ParamsConfig {
public:
  ParamsConfig() : type_(ConfigType::None) {}
  ParamsConfig(ConfigType type) : type_(type) {
    switch (type_) {
    case ConfigType::Algorithm:
      new (&algorithmConfig_) AlgorithmConfig();
      break;
    case ConfigType::Stream:
      new (&cameraConfig_) CameraConfig();
      break;
    case ConfigType::Logic:
      new (&logicConfig_) LogicConfig();
      break;
    case ConfigType::Output:
      new (&outputConfig_) OutputConfig();
      break;
    default:
      assert(false);
      break;
    }
  }

  ~ParamsConfig() {
    switch (type_) {
    case ConfigType::Algorithm:
      algorithmConfig_.~AlgorithmConfig();
      break;
    case ConfigType::Stream:
      cameraConfig_.~CameraConfig();
      break;
    case ConfigType::Logic:
      logicConfig_.~LogicConfig();
      break;
    case ConfigType::Output:
      outputConfig_.~OutputConfig();
      break;
    case ConfigType::None:
      break;
    default:
      assert(false);
      break;
    }
    type_ = ConfigType::None;
  }

  ParamsConfig(const ParamsConfig &other) : type_(other.type_) {
    switch (type_) {
    case ConfigType::None:
      break;
    case ConfigType::Algorithm:
      new (&algorithmConfig_) AlgorithmConfig(other.algorithmConfig_);
      break;
    case ConfigType::Stream:
      new (&cameraConfig_) CameraConfig(other.cameraConfig_);
      break;
    case ConfigType::Logic:
      new (&logicConfig_) LogicConfig(other.logicConfig_);
      break;
    case ConfigType::Output:
      new (&outputConfig_) OutputConfig(other.outputConfig_);
      break;
    default:
      assert(false);
      break;
    }
  }

  ParamsConfig(ParamsConfig &&other) : type_(other.type_) {
    switch (type_) {
    case ConfigType::None:
      break;
    case ConfigType::Algorithm:
      new (&algorithmConfig_)
          AlgorithmConfig(std::move(other.algorithmConfig_));
      break;
    case ConfigType::Stream:
      new (&cameraConfig_) CameraConfig(std::move(other.cameraConfig_));
      break;
    case ConfigType::Logic:
      new (&logicConfig_) LogicConfig(std::move(other.logicConfig_));
      break;
    case ConfigType::Output:
      new (&outputConfig_) OutputConfig(std::move(other.outputConfig_));
      break;
    default:
      assert(false);
      break;
    }
    other.type_ = ConfigType::None;
  }

  ParamsConfig &operator=(const ParamsConfig &other) {
    if (&other != this) {
      switch (other.type_) {
      case ConfigType::None:
        this->~ParamsConfig();
        break;
      case ConfigType::Algorithm:
        *this = other.algorithmConfig_;
        break;
      case ConfigType::Stream:
        *this = other.cameraConfig_;
        break;
      case ConfigType::Logic:
        *this = other.logicConfig_;
        break;
      case ConfigType::Output:
        *this = other.outputConfig_;
        break;
      default:
        assert(false);
        break;
      }
    }
    return *this;
  }

  ParamsConfig &operator=(ParamsConfig &&other) {
    assert(this != &other);
    switch (other.type_) {
    case ConfigType::None:
      this->~ParamsConfig();
      break;
    case ConfigType::Algorithm:
      *this = std::move(other.algorithmConfig_);
      break;
    case ConfigType::Stream:
      *this = std::move(other.cameraConfig_);
      break;
    case ConfigType::Logic:
      *this = std::move(other.logicConfig_);
      break;
    case ConfigType::Output:
      *this = std::move(other.outputConfig_);
      break;
    default:
      assert(false);
      break;
    }
    other.type_ = ConfigType::None;
    return *this;
  }

  ParamsConfig(const AlgorithmConfig &a)
      : type_(ConfigType::Algorithm), algorithmConfig_(a) {}

  ParamsConfig(AlgorithmConfig &&a)
      : type_(ConfigType::Algorithm), algorithmConfig_(std::move(a)) {}

  ParamsConfig &operator=(const AlgorithmConfig &a) {
    if (type_ != ConfigType::Algorithm) {
      this->~ParamsConfig();
      new (this) ParamsConfig(a);
    } else {
      algorithmConfig_ = a;
    }
    return *this;
  }

  ParamsConfig &operator=(AlgorithmConfig &&a) {
    if (type_ != ConfigType::Algorithm) {
      this->~ParamsConfig();
      new (this) ParamsConfig(std::move(a));
    } else {
      algorithmConfig_ = std::move(a);
    }
    return *this;
  }

  ParamsConfig(const CameraConfig &b)
      : type_(ConfigType::Stream), cameraConfig_(b) {}

  ParamsConfig(CameraConfig &&b)
      : type_(ConfigType::Stream), cameraConfig_(std::move(b)) {}

  ParamsConfig &operator=(const CameraConfig &b) {
    if (type_ != ConfigType::Stream) {
      this->~ParamsConfig();
      new (this) ParamsConfig(b);
    } else {
      cameraConfig_ = b;
    }
    return *this;
  }

  ParamsConfig &operator=(CameraConfig &&b) {
    if (type_ != ConfigType::Stream) {
      this->~ParamsConfig();
      new (this) ParamsConfig(std::move(b));
    } else {
      cameraConfig_ = std::move(b);
    }
    return *this;
  }

  ParamsConfig(const LogicConfig &b)
      : type_(ConfigType::Logic), logicConfig_(b) {}

  ParamsConfig(LogicConfig &&b)
      : type_(ConfigType::Logic), logicConfig_(std::move(b)) {}

  ParamsConfig &operator=(const LogicConfig &b) {
    if (type_ != ConfigType::Logic) {
      this->~ParamsConfig();
      new (this) ParamsConfig(b);
    } else {
      logicConfig_ = b;
    }
    return *this;
  }

  ParamsConfig &operator=(LogicConfig &&b) {
    if (type_ != ConfigType::Logic) {
      this->~ParamsConfig();
      new (this) ParamsConfig(std::move(b));
    } else {
      logicConfig_ = std::move(b);
    }
    return *this;
  }

  ParamsConfig(const OutputConfig &b)
      : type_(ConfigType::Output), outputConfig_(b) {}

  ParamsConfig(OutputConfig &&b)
      : type_(ConfigType::Output), outputConfig_(std::move(b)) {}

  ParamsConfig &operator=(const OutputConfig &b) {
    if (type_ != ConfigType::Output) {
      this->~ParamsConfig();
      new (this) ParamsConfig(b);
    } else {
      outputConfig_ = b;
    }
    return *this;
  }

  ParamsConfig &operator=(OutputConfig &&b) {
    if (type_ != ConfigType::Output) {
      this->~ParamsConfig();
      new (this) ParamsConfig(std::move(b));
    } else {
      outputConfig_ = std::move(b);
    }
    return *this;
  }

  ConfigType GetKind() const { return type_; }

  AlgorithmConfig &GetAlgorithmConfig() {
    assert(type_ == ConfigType::Algorithm);
    return algorithmConfig_;
  }

  const AlgorithmConfig &GetAlgorithmConfig() const {
    assert(type_ == ConfigType::Algorithm);
    return algorithmConfig_;
  }

  CameraConfig &GetCameraConfig() {
    assert(type_ == ConfigType::Stream);
    return cameraConfig_;
  }

  const CameraConfig &GetCameraConfig() const {
    assert(type_ == ConfigType::Stream);
    return cameraConfig_;
  }

  LogicConfig &GetLogicConfig() {
    assert(type_ == ConfigType::Logic);
    return logicConfig_;
  }

  const LogicConfig &GetLogicConfig() const {
    assert(type_ == ConfigType::Logic);
    return logicConfig_;
  }

  OutputConfig &GetOutputConfig() {
    assert(type_ == ConfigType::Output);
    return outputConfig_;
  }

  const OutputConfig &GetOutputConfig() const {
    assert(type_ == ConfigType::Output);
    return outputConfig_;
  }

private:
  ConfigType type_;
  union {
    // 摄像机参数
    CameraConfig cameraConfig_;

    // 算法参数
    AlgorithmConfig algorithmConfig_;

    // 逻辑模块参数
    LogicConfig logicConfig_;

    // 发送模块参数
    OutputConfig outputConfig_;
  };
};
} // namespace common

#endif // _FLOWENGINE_COMMON_CONFIG_HPP_