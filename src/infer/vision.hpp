/**
 * @file vision.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-07
 *
 * @copyright Copyright (c) 2022
 *
 */

/**
 * @file inference.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-01
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __INFER_VISION_H_
#define __INFER_VISION_H_
#include "infer_common.hpp"

namespace infer {
namespace vision {

class Vision {
public:
  Vision(common::AlgorithmConfig const &_param) : mParams(_param) {}

  virtual ~Vision(){};

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processInput(void *) const = 0;

  //!
  //! \brief ProcessInput that the input is correct for infer
  //!
  virtual bool processOutput(void *, Result &) const = 0;

  //!
  //! \brief verifyOutput that the result is correct for infer
  //!
  virtual bool verifyOutput(Result const &) const = 0;

protected:
  //!< The parameters for the sample.
  common::AlgorithmConfig mParams;
};
} // namespace vision
} // namespace infer
#endif
