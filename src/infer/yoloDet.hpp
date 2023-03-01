/**
 * @file yoloDet.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-07
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __INFERENCE_VISION_YOLO_DETECTION_H_
#define __INFERENCE_VISION_YOLO_DETECTION_H_
#include "detection.hpp"
#include <vector>

namespace infer {
namespace vision {

class Yolo : public Detection {
  //!
  //! \brief construction
  //!
public:
  Yolo(const AlgoConfig &_param, ModelInfo const &info)
      : Detection(_param, info) {}

private:
  virtual void generateBoxes(std::unordered_map<int, BBoxes> &,
                             void **) const override;
};
} // namespace vision
} // namespace infer

#endif