/**
 * @file detection.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-03
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __INFERENCE_DETECTION_H_
#define __INFERENCE_DETECTION_H_
#include "jetson/trt_inference.hpp"

namespace infer {
namespace trt {
class DetectionInfer : public TRTInference {
  //!
  //! \brief construction
  //!
public:
  DetectionInfer(const common::AlgorithmConfig &_param)
      : TRTInference(_param) {}

protected:
  //!
  //! \brief Postprocessing that the output is correct and prints it
  //!
  virtual bool processOutput(BufferManager const &, Result &) const override;

  static inline bool compare(DetectionResult const &a,
                             DetectionResult const &b) {
    return a.det_confidence > b.det_confidence;
  }

  float iou(std::array<float, 4> const &, std::array<float, 4> const &) const;

  virtual void
  generateBoxes(std::unordered_map<int, std::vector<DetectionResult>> &,
                BufferManager const &) const = 0;

  virtual void
  nms(std::vector<DetectionResult> &,
      std::unordered_map<int, std::vector<DetectionResult>> &) const;

  virtual void renderOriginShape(std::vector<DetectionResult> &results,
                                 std::array<int, 3> const &shape) const;

  int DETECTION_SIZE = sizeof(DetectionResult) / sizeof(float);
};
} // namespace trt
} // namespace infer

#endif