#include <array>
#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifndef _FLOWENGINE_COMMON_CONFIG_HPP_
#define _FLOWENGINE_COMMON_CONFIG_HPP_

namespace common {

enum class ModuleType { Detection = 0, Classifier, Stream, Output, None };

enum class WorkerTypes { Calling, CatDog, Smoking };

// 接收管理进程给出发生变化的配置
struct FlowConfigure {
  bool status; // 只有启动和关闭两种状态 true 启动，false 关闭
  int hostId;  // 设备主机id
  int provinceId; // 省
  int cityId;     // 市
  int regionId;   // 区
  int stationId;  // 站
  int location;   // 站内的那个区域（卸油区、加油区等等）
  std::string cameraId;  // 摄像机唯一ID
  std::string alarmType; // 报警类型（smoke, phone等等）
  std::string videoCode; // 视频编码类型
  std::string flowType;  // 流协议类型
  std::string cameraIp;  // 网络流链接
};

struct ModuleConfigure {
  std::string moduleName; // 模块名称
  std::string sendName;   // 下游模块
  std::string recvName;   // 上游模块
};

struct AlgorithmConfig {
  AlgorithmConfig() = default;
  AlgorithmConfig(ModuleType type_, std::string const &modelPath_,
                  std::vector<std::string> const &inputTensorNames_,
                  std::vector<std::string> const &outputTensorNames_,
                  std::array<int, 3> const &inputShape_, int numClasses_,
                  int numAnchors_ = 0,
                  std::array<int, 4> region_ = {0, 0, 0, 0},
                  float cond_thr_ = 0.3, float nms_thr_ = 0.5,
                  int batchSize_ = 1)
      : type(type_), modelPath(modelPath_), numClasses(numClasses_),
        inputTensorNames(std::move(inputTensorNames_)),
        outputTensorNames(std::move(outputTensorNames_)),
        inputShape(std::move(inputShape_)), region(region_),
        cond_thr(cond_thr_), numAnchors(numAnchors_), nms_thr(nms_thr_),
        batchSize(batchSize_){};
  // 算法配置
  ModuleType type;
  std::string modelPath;                      // engine 所在目录
  int numClasses;                             // classes of number
  std::vector<std::string> inputTensorNames;  // input tensor names
  std::vector<std::string> outputTensorNames; // output tensor names
  std::array<int, 3> inputShape;              // 算法需要的输入尺度
  std::array<int, 4> region;                  // 划定区域
  float cond_thr;                             // 置信度阈值
  float nms_thr;                              // NMS 阈值
  int numAnchors; // detection model anchors of number  yolov5s: 25200
  int batchSize;  // batch of number
};

struct CameraConfig {
  CameraConfig(int widthPixel_, int heightPixel_, std::string const &cameraId_,
               std::string const &videoCode_, std::string const &flowType_,
               std::string const &cameraIp_)
      : widthPixel(widthPixel_), heightPixel(heightPixel_), cameraId(cameraId_),
        videoCode(videoCode_), flowType(flowType_), cameraIp(cameraIp_) {}
  CameraConfig() = default;
  ~CameraConfig() {}
  // 摄像机配置
  int widthPixel;        // 视频宽度
  int heightPixel;       // 视频高度
  std::string cameraId;  // 摄像机唯一ID
  std::string videoCode; // 视频编码类型
  std::string flowType;  // 流协议类型
  std::string cameraIp;  // 网络流链接
};

struct SendConfig {
  SendConfig() = default;
  SendConfig(std::string const &url_) : url(url_) {}
  ~SendConfig() {}
  std::string url;
};

class ParamsConfig {
public:
  ParamsConfig() : type_(ModuleType::None) {}
  ParamsConfig(ModuleType type) : type_(type) {
    switch (type_) {
    case ModuleType::Classifier:
    case ModuleType::Detection:
      new (&algorithmConfig_) AlgorithmConfig();
      break;
    case ModuleType::Stream:
      new (&cameraConfig_) CameraConfig();
      break;
    case ModuleType::Output:
      new (&sendConfig_) SendConfig();
      break;
    default:
      assert(false);
      break;
    }
  }

  ~ParamsConfig() {
    switch (type_) {
    case ModuleType::Classifier:
    case ModuleType::Detection:
      algorithmConfig_.~AlgorithmConfig();
      break;
    case ModuleType::Stream:
      cameraConfig_.~CameraConfig();
      break;
    case ModuleType::Output:
      sendConfig_.~SendConfig();
      break;
    default:
      assert(false);
      break;
    }
    type_ = ModuleType::None;
  }

  ParamsConfig(const ParamsConfig &other) : type_(other.type_) {
    switch (type_) {
    case ModuleType::None:
      break;
    case ModuleType::Classifier:
    case ModuleType::Detection:
      new (&algorithmConfig_) AlgorithmConfig(other.algorithmConfig_);
      break;
    case ModuleType::Stream:
      new (&cameraConfig_) CameraConfig(other.cameraConfig_);
      break;
    case ModuleType::Output:
      new (&sendConfig_) SendConfig(other.sendConfig_);
      break;
    default:
      assert(false);
      break;
    }
  }

  ParamsConfig(ParamsConfig &&other) : type_(other.type_) {
    switch (type_) {
    case ModuleType::None:
      break;
    case ModuleType::Classifier:
    case ModuleType::Detection:
      new (&algorithmConfig_)
          AlgorithmConfig(std::move(other.algorithmConfig_));
      break;
    case ModuleType::Stream:
      new (&cameraConfig_) CameraConfig(std::move(other.cameraConfig_));
      break;
    case ModuleType::Output:
      new (&sendConfig_) SendConfig(std::move(other.sendConfig_));
      break;
    default:
      assert(false);
      break;
    }
    other.type_ = ModuleType::None;
  }

  ParamsConfig &operator=(const ParamsConfig &other) {
    if (&other != this) {
      switch (other.type_) {
      case ModuleType::None:
        this->~ParamsConfig();
        break;
      case ModuleType::Classifier:
      case ModuleType::Detection:
        *this = other.algorithmConfig_;
        break;
      case ModuleType::Stream:
        *this = other.cameraConfig_;
        break;
      case ModuleType::Output:
        *this = other.sendConfig_;
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
    case ModuleType::None:
      this->~ParamsConfig();
      break;
    case ModuleType::Classifier:
    case ModuleType::Detection:
      *this = std::move(other.algorithmConfig_);
      break;
    case ModuleType::Stream:
      *this = std::move(other.cameraConfig_);
      break;
    case ModuleType::Output:
      *this = std::move(other.sendConfig_);
      break;
    default:
      assert(false);
      break;
    }
    other.type_ = ModuleType::None;
    return *this;
  }

  ParamsConfig(const AlgorithmConfig &a) : type_(a.type), algorithmConfig_(a) {}

  ParamsConfig(AlgorithmConfig &&a)
      : type_(a.type), algorithmConfig_(std::move(a)) {}

  ParamsConfig &operator=(const AlgorithmConfig &a) {
    if (type_ != a.type) {
      this->~ParamsConfig();
      new (this) ParamsConfig(a);
    } else {
      algorithmConfig_ = a;
    }
    return *this;
  }

  ParamsConfig &operator=(AlgorithmConfig &&a) {
    if (type_ != a.type) {
      this->~ParamsConfig();
      new (this) ParamsConfig(std::move(a));
    } else {
      algorithmConfig_ = std::move(a);
    }
    return *this;
  }

  ParamsConfig(const CameraConfig &b)
      : type_(ModuleType::Stream), cameraConfig_(b) {}

  ParamsConfig(CameraConfig &&b)
      : type_(ModuleType::Stream), cameraConfig_(std::move(b)) {}

  ParamsConfig &operator=(const CameraConfig &b) {
    if (type_ != ModuleType::Stream) {
      this->~ParamsConfig();
      new (this) ParamsConfig(b);
    } else {
      cameraConfig_ = b;
    }
    return *this;
  }

  ParamsConfig &operator=(CameraConfig &&b) {
    if (type_ != ModuleType::Stream) {
      this->~ParamsConfig();
      new (this) ParamsConfig(std::move(b));
    } else {
      cameraConfig_ = std::move(b);
    }
    return *this;
  }

  ParamsConfig(const SendConfig &b)
      : type_(ModuleType::Output), sendConfig_(b) {}

  ParamsConfig(SendConfig &&b)
      : type_(ModuleType::Output), sendConfig_(std::move(b)) {}

  ParamsConfig &operator=(const SendConfig &b) {
    if (type_ != ModuleType::Output) {
      this->~ParamsConfig();
      new (this) ParamsConfig(b);
    } else {
      sendConfig_ = b;
    }
    return *this;
  }

  ParamsConfig &operator=(SendConfig &&b) {
    if (type_ != ModuleType::Output) {
      this->~ParamsConfig();
      new (this) ParamsConfig(std::move(b));
    } else {
      sendConfig_ = std::move(b);
    }
    return *this;
  }

  ModuleType GetKind() const { return type_; }

  AlgorithmConfig &GetAlgorithmConfig() {
    assert(type_ == ModuleType::Classifier || type_ == ModuleType::Detection);
    return algorithmConfig_;
  }

  const AlgorithmConfig &GetAlgorithmConfig() const {
    assert(type_ == ModuleType::Classifier || type_ == ModuleType::Detection);
    return algorithmConfig_;
  }

  CameraConfig &GetCameraConfig() {
    assert(type_ == ModuleType::Stream);
    return cameraConfig_;
  }

  const CameraConfig &GetCameraConfig() const {
    assert(type_ == ModuleType::Stream);
    return cameraConfig_;
  }

  SendConfig &GetSendConfig() {
    assert(type_ == ModuleType::Output);
    return sendConfig_;
  }

  const SendConfig &GetSendConfig() const {
    assert(type_ == ModuleType::Output);
    return sendConfig_;
  }

private:
  ModuleType type_;
  union {
    // 摄像机参数
    CameraConfig cameraConfig_;

    // 算法参数
    AlgorithmConfig algorithmConfig_;

    // 发送结果的参数
    SendConfig sendConfig_;
  };
};
} // namespace common

// int main() {
// A a(1, "Hello from A");
// B b(2, "Hello from B");

// ParamsConfig mv_1 = a;

// cout << "mv_1 = a: " << mv_1.GetA().name << endl;
// mv_1 = b;
// cout << "mv_1 = b: " << mv_1.GetB().name << endl;
// mv_1 = A(3, "hello again from A");
// cout << R"aaa(mv_1 = A(3, "hello again from A"): )aaa" << mv_1.GetA().name
//      << endl;
// mv_1 = 42;
// cout << "mv_1 = 42: " << mv_1.GetInteger() << endl;

// b.vec = {10, 20, 30, 40, 50};

// mv_1 = move(b);
// cout << "After move, mv_1 = b: vec.size = " << mv_1.GetB().vec.size() <<
// endl;

// cout << endl << "Press a letter" << endl;
// char c;
// cin >> c;
// }

#endif // _FLOWENGINE_COMMON_CONFIG_HPP_