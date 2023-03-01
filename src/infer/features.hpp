/**
 * @file pose.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-22
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#ifndef __INFERENCE_VISION_FEATURES_H_
#define __INFERENCE_VISION_FEATURES_H_
#include "core/factory.hpp"
#include "vision.hpp"
#include <unordered_map>
#include <vector>

namespace infer {
namespace vision {

class Features : public Vision {
  //!
  //! \brief construction
  //!
public:
  Features(const AlgoConfig &_param, ModelInfo const &_info)
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
  virtual bool processOutput(void **, InferResult &) const override;

  //!
  //! \brief verifyOutput that the result is correct for infer
  //!
  virtual bool verifyOutput(InferResult const &) const override;

};
} // namespace vision
} // namespace infer

#endif