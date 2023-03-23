/**
 * @file lprDet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __INFERENCE_VISION_LPR_DETECTION_H_
#define __INFERENCE_VISION_LPR_DETECTION_H_
#include "detection.hpp"
#include <vector>

namespace infer {
namespace vision {

class LPRDet : public Detection {
  //!
  //! \brief construction
  //!
public:
  LPRDet(const AlgoConfig &_param, ModelInfo const &info)
      : Detection(_param, info) {}

private:
  virtual bool processOutput(void **, InferResult &) const override;

  virtual void generateBoxes(std::unordered_map<int, BBoxes> &,
                             void **) const override;

  void generateKeypointsBoxes(std::unordered_map<int, KeypointsBoxes> &,
                              void **) const;
};
} // namespace vision
} // namespace infer

#endif