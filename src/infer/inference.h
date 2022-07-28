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
#ifndef __INFERENCE_H_
#define __INFERENCE_H_
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include "common/common.hpp"

namespace infer {

struct alignas(float) DetectionResult {
  // x y w h
  std::array<float, 4> bbox;  // [x1, y1, x2, y2]
  float det_confidence;
  float class_id;
  float class_confidence;
};

struct Result {
  std::vector<DetectionResult> detResults;
  std::pair<int, float> classResult;
  std::array<int, 3> shape;
};

class Inference {
public:
  virtual ~Inference() {}
  //!
  //! \brief initialize the network
  //!
  virtual bool initialize() = 0;

  //!
  //! \brief Runs the inference engine with input of Mat
  //!
  virtual bool infer(void*, Result &) = 0;
};
} // namespace infer
#endif
