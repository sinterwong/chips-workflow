/**
 * @file visionInfer.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "core/algoInfer.hpp"
#include "vision.hpp"
#if (TARGET_PLATFORM == 0)
#include "x3/x3_inference.hpp"
using namespace infer::x3;
#elif (TARGET_PLATFORM == 1)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
#elif (TARGET_PLATFORM == 2)
#include "jetson/trt_inference.hpp"
using namespace infer::trt;
#endif

namespace infer {
using common::AlgorithmConfig;
using common::InferResult;

class VisionInfer : public AlgoInfer {
public:
  VisionInfer(AlgorithmConfig const& config_) : AlgoInfer(config_) {}
  /**
   * @brief 初始化算法
   *
   * @return true
   * @return false
   */
  virtual bool init() override;

  virtual bool infer(void* data, InferParams const &, InferResult &ret) override;

private:
  std::mutex m;
  std::shared_ptr<AlgoInference> instance;
  std::shared_ptr<vision::Vision> vision;
  infer::ModelInfo modelInfo;
};
} // namespace infer