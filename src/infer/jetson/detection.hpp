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
#ifndef __TRT_INFERENCE_DETECTION_H_
#define __TRT_INFERENCE_DETECTION_H_
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

  virtual void
  generateBoxes(std::unordered_map<int, std::vector<DetectionResult>> &,
                BufferManager const &) const = 0;
};
} // namespace trt
} // namespace infer

#endif