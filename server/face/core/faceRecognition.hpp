/**
 * @file faceRecognition.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 人脸识别逻辑串联，输入单帧图像，输出最中心人脸特征
 * @version 0.1
 * @date 2023-10-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <atomic>
#include <opencv2/core/mat.hpp>

#ifndef __SERVER_FACE_CORE_FACE_RECOGNITION_HPP_
#define __SERVER_FACE_CORE_FACE_RECOGNITION_HPP_
namespace server::face::core {

class FaceRecognition {
public:
  FaceRecognition() {}
  ~FaceRecognition() {}

  bool forward(cv::Mat &image, std::vector<float> &feature) {

    // to do something

    return true;
  }

private:
  std::atomic_bool status = false;
};
} // namespace server::face::core
#endif