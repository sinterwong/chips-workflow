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

#ifndef __INFERENCE_VISION_CRNN_CTC_REC_H_
#define __INFERENCE_VISION_CRNN_CTC_REC_H_
#include "charsRec.hpp"

namespace infer::vision {

class CRNN : public CharsRec {
  //!
  //! \brief construction
  //!
public:
  explicit CRNN(const AlgoConfig &_param, ModelInfo const &info)
      : CharsRec(_param, info) {}

private:
  CharsRet generateChars(void **output) const override;

  CharsRet decodeChars(CharsRet const &) const;
};
} // namespace infer::vision

#endif