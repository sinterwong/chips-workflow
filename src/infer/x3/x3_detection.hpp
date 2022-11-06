/**
 * @file x3_detection.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFERENCE_X3_DETECTION_H_
#define __INFERENCE_X3_DETECTION_H_
#include "x3_inference.hpp"
#include <vector>

namespace infer {
namespace x3 {
class X3DetectionInfer : public X3Inference {
  //!
  //! \brief construction
  //!
public:
  X3DetectionInfer(const common::AlgorithmConfig &_param)
      : X3Inference(_param) {}

protected:
  //!
  //! \brief Postprocessing that the output is correct and prints it
  //!
  virtual bool processOutput(void *, Result &) const override;

  virtual void
  generateBoxes(std::unordered_map<int, std::vector<DetectionResult>> &,
                void *) const = 0;
};
} // namespace x3
} // namespace infer

#endif