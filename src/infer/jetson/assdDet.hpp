/**
 * @file assdDet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-28
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __INFERENCE_ASSD_DETECTION_H_
#define __INFERENCE_ASSD_DETECTION_H_
#include "jetson/detection.hpp"

namespace infer {
namespace trt {
class AssdDet : public DetectionInfer {
  //!
  //! \brief construction
  //!
public:
  AssdDet(const common::AlgorithmConfig &_param) : DetectionInfer(_param) {}

private:
  //!
  //! \brief Verifies that the output is correct and prints it
  //!
  virtual bool verifyOutput(Result const &) const override;

  //!
  //! \brief Boxes generates rules
  //!
  virtual void
  generateBoxes(std::unordered_map<int, std::vector<DetectionResult>> &,
                BufferManager const &) const override;

  // std::vector<float> receptive_field_center_start = {7, 7, 7};
  // std::vector<float> receptive_field_stride = {8, 8, 8};
  // std::vector<float> RF_half = {55.5, 71.5, 79.5};
  std::vector<float> receptive_field_center_start = {12, 20, 28, 36};
  std::vector<float> receptive_field_stride = {4, 8, 8, 8};
  std::vector<float> RF_half = {38.5, 45.5, 55.5, 79.5};
};
} // namespace trt
} // namespace infer

#endif