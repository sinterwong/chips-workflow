/**
 * @file faceNet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-04-03
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __INFERENCE_VISION_FACENET_RECO_H_
#define __INFERENCE_VISION_FACENET_RECO_H_
#include "features.hpp"

namespace infer::vision {

class FaceNet : public Features {
  //!
  //! \brief construction
  //!
public:
  FaceNet(const AlgoConfig &_param, ModelInfo const &info)
      : Features(_param, info) {}

private:
  virtual void generateFeature(void **output, Eigenvector &feature) const override;
};
} // namespace infer::vision

#endif