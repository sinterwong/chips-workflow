/**
 * @file detection.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFERENCE_VISION_DETECTION_H_
#define __INFERENCE_VISION_DETECTION_H_
#include "common/factory.hpp"
#include "vision.hpp"
#include <memory>
#include <unordered_map>
#include <vector>

namespace infer::vision {
class Detection : public Vision {
  //!
  //! \brief construction
  //!
public:
  explicit Detection(const AlgoConfig &_param, ModelInfo const &_info)
      : Vision(_param, _info) {
    config = mParams.getCopyParams<DetAlgo>();
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
  DetAlgo config;
  virtual void generateBoxes(std::unordered_map<int, BBoxes> &,
                             void **) const = 0;
};
} // namespace infer::vision

#endif