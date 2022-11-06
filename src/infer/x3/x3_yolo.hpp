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

#ifndef __INFERENCE_X3_YOLO_DETECTION_H_
#define __INFERENCE_X3_YOLO_DETECTION_H_
#include "x3_detection.hpp"

namespace infer {
namespace x3 {
class YoloDet : public X3DetectionInfer {
  //!
  //! \brief construction
  //!
public:
  YoloDet(const common::AlgorithmConfig &_param) : X3DetectionInfer(_param) {}

private:
  //!
  //! \brief Verifies that the output is correct and prints it
  //!
  virtual bool verifyOutput(Result const &) const override;

  virtual void
  generateBoxes(std::unordered_map<int, std::vector<DetectionResult>> &,
                void *) const override;
};
} // namespace x3
} // namespace infer

#endif