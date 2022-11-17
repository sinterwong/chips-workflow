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
#include "core/factory.hpp"
#include "vision.hpp"
#include <unordered_map>
#include <vector>

namespace infer {
namespace vision {

class Detection : public Vision {
  //!
  //! \brief construction
  //!
public:
  Detection(const common::AlgorithmConfig &_param, ModelInfo const &_info)
      : Vision(_param, _info) {}

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processInput(cv::Mat const &input, void **output,
                            common::ColorType,
                            common::ColorType) const override;

  //!
  //! \brief Postprocessing that the output is correct and prints it
  //!
  virtual bool processOutput(void **, Result &) const override;

  //!
  //! \brief verifyOutput that the result is correct for infer
  //!
  virtual bool verifyOutput(Result const &) const override;

protected:
  virtual void
  generateBoxes(std::unordered_map<int, std::vector<DetectionResult>> &,
                void **) const = 0;
};
} // namespace vision
} // namespace infer

#endif