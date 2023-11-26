/**
 * @file faceKeyPoints.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-23
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "keypoints.hpp"
#include <vector>

#ifndef __INFERENCE_VISION_FACE_KEYPOINTS_DETECTION_H_
#define __INFERENCE_VISION_FACE_KEYPOINTS_DETECTION_H_
namespace infer::vision {

class FaceKeyPoints : public Keypoints {
  //!
  //! \brief construction
  //!
public:
  explicit FaceKeyPoints(const AlgoConfig &_param, ModelInfo const &info)
      : Keypoints(_param, info) {}

private:
  virtual bool processOutput(void **, InferResult &) const override;

  virtual void generateKeypoints(KeypointsRet &, void **) const override;

private:
  void transPoints2d(std::vector<Point2f> &pts, cv::Mat const &M) const;
};
} // namespace infer::vision

#endif