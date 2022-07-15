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
#include "inference.h"
#include "jetson/trt_inference.hpp"

namespace infer {
namespace trt {
class DetctionInfer : public TRTInference {
  //!
  //! \brief construction
  //!
public:
  DetctionInfer(InferParams const &params) : TRTInference(params) {}

private:
  //!
  //! \brief Verifies that the output is correct and prints it
  //!
  virtual bool verifyOutput(Result const &) const override;

  //!
  //! \brief Postprocessing that the output is correct and prints it
  //!
  virtual bool processOutput(BufferManager const &,
                             Result &) const override;

  static inline bool compare(DetectionResult const &a,
                             DetectionResult const &b) {
    return a.det_confidence > b.det_confidence;
  }

  float iou(std::array<float, 4> const &, std::array<float, 4> const &) const;

  void nms(std::vector<DetectionResult> &, float *) const;
  int DETECTION_SIZE = sizeof(DetectionResult) / sizeof(float);
};
} // namespace trt
} // namespace infer

#endif