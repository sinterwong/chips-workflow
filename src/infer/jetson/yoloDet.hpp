/**
 * @file yoloDet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-28
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFERENCE_YOLO_DETECTION_H_
#define __INFERENCE_YOLO_DETECTION_H_
#include "jetson/detection.hpp"

namespace infer {
namespace trt {
class YoloDet : public DetectionInfer {
  //!
  //! \brief construction
  //!
public:
  YoloDet(const common::AlgorithmConfig &_param)
      : DetectionInfer(_param) {}

private:
  //!
  //! \brief Verifies that the output is correct and prints it
  //!
  virtual bool verifyOutput(Result const &) const override;

  virtual void
  generateBoxes(std::unordered_map<int, std::vector<DetectionResult>> &,
                BufferManager const &) const override;
};
} // namespace trt
} // namespace infer

#endif