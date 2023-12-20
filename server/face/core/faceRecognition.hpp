/**
 * @file faceRecognition.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 人脸识别逻辑串联，输入单帧图像，输出最中心人脸特征
 * @version 0.2
 * @date 2023-11-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <atomic>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "algoManager.hpp"

#ifndef __SERVER_FACE_CORE_FACE_RECOGNITION_HPP_
#define __SERVER_FACE_CORE_FACE_RECOGNITION_HPP_
namespace server::face::core {

class FaceRecognition {
public:
  static FaceRecognition &getInstance() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] { instance = new FaceRecognition(); });
    return *instance;
  }
  FaceRecognition(FaceRecognition const &) = delete;
  FaceRecognition &operator=(FaceRecognition const &) = delete;

public:

  bool extract(FramePackage const &framePackage, std::vector<float> &feature);

  bool extract(std::string const &url, std::vector<float> &feature);

private:
  FaceRecognition() {}
  ~FaceRecognition() {
    delete instance;
    instance = nullptr;
  }
  static FaceRecognition *instance;

private:
  cv::Mat estimateNorm(const std::vector<cv::Point2f> &landmarks,
                       int imageSize = 112);

  cv::Mat normCrop(const cv::Mat &img,
                   const std::vector<cv::Point2f> &landmarks,
                   int imageSize = 112);

  void getFaceInput(cv::Mat const &input, cv::Mat &output, FrameInfo &frame,
                    Points2f const &points, ColorType const &type);
};
} // namespace server::face::core
#endif
