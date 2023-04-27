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
#include "detection.hpp"

namespace infer::vision {
class Assd : public Detection {
  //!
  //! \brief construction
  //!
public:
  explicit Assd(const AlgoConfig &_param, ModelInfo const &info)
      : Detection(_param, info) {}

private:
  //!
  //! \brief Verifies that the output is correct and prints it
  //!
  virtual bool verifyOutput(InferResult const &) const override;

  //!
  //! \brief Boxes generates rules
  //!
  virtual void generateBoxes(std::unordered_map<int, BBoxes> &,
                             void **) const override;

  std::vector<float> receptive_field_center_start = {7, 7, 7};
  std::vector<float> receptive_field_stride = {8, 8, 8};
  std::vector<float> RF_half = {55.5, 71.5, 79.5};
};
} // namespace infer::vision

#endif