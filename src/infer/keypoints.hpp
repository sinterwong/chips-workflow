/**
 * @file keypoints.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-04-04
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "common/factory.hpp"
#include "vision.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

#ifndef __INFERENCE_VISION_DETECTION_H_
#define __INFERENCE_VISION_DETECTION_H_
namespace infer::vision {
class Keypoints : public Vision {
  //!
  //! \brief construction
  //!
public:
  explicit Keypoints(const AlgoConfig &_param, ModelInfo const &_info)
      : Vision(_param, _info) {
    config = mParams.getCopyParams<PointsDetAlgo>();
  }

  //!
  //! \brief Postprocessing that the output is correct and prints it
  //!
  virtual bool processOutput(void **, InferResult &) const override;

  //!
  //! \brief verifyOutput that the result is correct for infer
  //!
  virtual bool verifyOutput(InferResult const &) const override;

protected:
  PointsDetAlgo config;
  virtual void generateKeypointsBoxes(std::unordered_map<int, KeypointsBoxes> &,
                                      void **) const {};

  virtual void generateKeypoints(KeypointsRet &, void **) const {};
};
} // namespace infer::vision

#endif