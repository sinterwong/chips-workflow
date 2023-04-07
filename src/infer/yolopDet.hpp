/**
 * @file YoloPDet.hpp
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
#include "keypoints.hpp"
#include <vector>

namespace infer::vision {

class YoloPDet : public Keypoints {
  //!
  //! \brief construction
  //!
public:
  YoloPDet(const AlgoConfig &_param, ModelInfo const &info)
      : Keypoints(_param, info) {}

private:
  virtual bool processOutput(void **, InferResult &) const override;

  virtual void generateKeypointsBoxes(std::unordered_map<int, KeypointsBoxes> &,
                                      void **) const override;
};
} // namespace infer::vision

#endif