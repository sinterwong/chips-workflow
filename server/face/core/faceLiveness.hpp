/**
 * @file faceLiveness.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __SERVER_FACE_CORE_FACE_LIVENESS_HPP_
#define __SERVER_FACE_CORE_FACE_LIVENESS_HPP_

#include "algoManager.hpp"
#include <mutex>

namespace server::face::core {

class FaceLiveness {
public:
  static FaceLiveness &getInstance() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] { instance = new FaceLiveness(); });
    return *instance;
  }

  FaceLiveness(FaceLiveness const &) = delete;
  FaceLiveness &operator=(FaceLiveness const &) = delete;

public:
  bool infer(FramePackage const &framePackage, int &liveness);

private:
  FaceLiveness() {}
  ~FaceLiveness() {
    delete instance;
    instance = nullptr;
  }
  static FaceLiveness *instance;

private:
  cv::Mat transform(int size, cv::Point2f center, double scale,
                    double rotation);

  cv::Mat normCrop(cv::Mat const &image, cv::Mat const &rotMat, int size);

  void getFaceInput(cv::Mat const &input, cv::Mat &output, FrameInfo &frame,
                    Points2f const &points, ColorType const &type);
};

} // namespace server::face::core

#endif // __SERVER_FACE_CORE_FACE_LIVENESS_HPP_