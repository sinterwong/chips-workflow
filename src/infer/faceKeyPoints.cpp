/**
 * @file faceKeyPoints.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-23
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "faceKeyPoints.hpp"
#include "logger/logger.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>

namespace infer::vision {

void FaceKeyPoints::transPoints2d(std::vector<Point2f> &pts,
                                  cv::Mat const &M) const {
  for (size_t i = 0; i < pts.size(); ++i) {
    cv::Mat ptMat = (cv::Mat_<float>(3, 1) << pts[i].x, pts[i].y, 1.0f);
    cv::Mat newPtMat = M * ptMat;

    pts[i].x = newPtMat.at<float>(0, 0);
    pts[i].y = newPtMat.at<float>(1, 0);
  }
}

bool FaceKeyPoints::processOutput(void **output, InferResult &result) const {

  auto keyPoints = std::get_if<KeypointsRet>(&result.aRet);
  if (keyPoints == nullptr) {
    FLOWENGINE_LOGGER_ERROR("Failed to get keyPointsRet");
    return false;
  }
  // 获取关键点
  generateKeypoints(*keyPoints, output);

  // 还原原始图像尺寸
  auto M = cv::Mat(2, 3, CV_32FC1, keyPoints->M);
  std::cout << M << std::endl;
  cv::Mat IM;
  // 旋转
  cv::invertAffineTransform(M, IM);

  // 处理结果
  transPoints2d(keyPoints->points, IM);
  return true;
}

void FaceKeyPoints::generateKeypoints(KeypointsRet &kps, void **outputs) const {
  float **output = reinterpret_cast<float **>(outputs);
  float *out = output[0]; // just one output
  int dim = modelInfo.outputShapes[0].at(1);

  for (int i = 0; i < dim; i += 2) {
    // 对每个值进行+1并且乘以输入尺寸的一半，得到原图上的坐标。因为输出区间是[-1,1]，所以要+1。
    out[i] = (out[i] + 1) * config.inputShape[0] / 2;         // x
    out[i + 1] = (out[i + 1] + 1) * config.inputShape[1] / 2; // y
    kps.points.emplace_back(Point2f{out[i], out[i + 1]});
  }
}

FlowEngineModuleRegister(FaceKeyPoints, AlgoConfig const &, ModelInfo const &);

} // namespace infer::vision