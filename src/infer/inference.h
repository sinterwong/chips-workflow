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

namespace infer {

struct alignas(float) DetectionResult {
  // x y w h
  std::array<float, 4> bbox;
  float det_confidence;
  float class_id;
  float class_confidence;
};

struct Result {
  std::vector<std::vector<DetectionResult>> detResults;
  int classResult;
};

struct InferParams{
  int32_t batchSize; //!< Number of inputs in a batch
  std::vector<std::string> inputTensorNames;
  std::vector<std::string> outputTensorNames;
  std::string serializedFilePath;
  short numClasses;
  int numAnchors;
  std::array<int, 3> inputShape;
  std::array<int, 3> originShape;
  float c_thr = 0.3;
  float nms_thr = 0.5;
  float scaling = 0.0;
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
  virtual void infer(void*, Result &) = 0;
};
} // namespace infer
#endif
