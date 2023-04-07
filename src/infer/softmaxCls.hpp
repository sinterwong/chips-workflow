/**
 * @file softmaxCls.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-02-20
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __INFERENCE_VISION_CROSS_ENTROPY_CLS_H_
#define __INFERENCE_VISION_CROSS_ENTROPY_CLS_H_
#include "classifier.hpp"

namespace infer::vision {

class Softmax : public Classifier {
  //!
  //! \brief construction
  //!
public:
  Softmax(const AlgoConfig &_param, ModelInfo const &info)
      : Classifier(_param, info) {}

private:
  ClsRet generateClass(void **output) const override;
};
} // namespace infer::vision

#endif