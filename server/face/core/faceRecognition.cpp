/**
 * @file faceRecognition.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-23
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "faceRecognition.hpp"
#include "faceInferUtils.hpp"
#include "networkUtils.hpp"
#include "postprocess.hpp"

namespace server::face::core {

FaceRecognition *FaceRecognition::instance = nullptr;

bool FaceRecognition::extract(FramePackage const &framePackage,
                              std::vector<float> &feature) {

  // 使用算法资源进行推理
  FrameInfo frame;
  frame.data = reinterpret_cast<void **>(&framePackage.frame->data);
  frame.inputShape = {framePackage.frame->cols, framePackage.frame->rows,
                      framePackage.frame->channels()};

  frame.shape = getInferShape(framePackage.frame->cols, framePackage.frame->rows);
  frame.type = getPlatformColorType();

  // 人脸检测
  InferResult faceDetRet;
  auto ret = AlgoManager::getInstance().infer(AlgoType::DET, frame, faceDetRet);
  if (!ret.get()) {
    FLOWENGINE_LOGGER_ERROR("Face detection failed!");
    return false;
  }
  auto kbboxes = std::get_if<KeypointsBoxes>(&faceDetRet.aRet);
  if (!kbboxes || kbboxes->empty()) {
    FLOWENGINE_LOGGER_INFO("Not a single face was detected!");
    return false;
  }

  // 获取最靠近中心的人脸
  size_t index = utils::findClosestBBoxIndex(*kbboxes, framePackage.frame->cols,
                                             framePackage.frame->rows);

  auto kbbox = kbboxes->at(index);

  // 获取人脸识别输入图像
  FrameInfo faceInput;
  cv::Mat faceImage;
  getFaceInput(*framePackage.frame, faceImage, faceInput, kbbox.points,
               frame.type);
  // cv::imwrite("output.jpg", faceImage);
  // 人脸特征提取
  InferResult faceRecRet;
  ret = AlgoManager::getInstance().infer(AlgoType::REC, faceInput, faceRecRet);
  if (!ret.get()) {
    FLOWENGINE_LOGGER_ERROR("Face recognition failed!");
    return false;
  }
  feature = *std::get_if<std::vector<float>>(&faceRecRet.aRet);
  utils::normalize_L2(feature.data(), feature.size());
  return true;
}

bool FaceRecognition::extract(std::string const &url,
                              std::vector<float> &feature) {
  // 推理任务定义
  std::shared_ptr<cv::Mat> image_bgr = getImageByUri(url);
  if (image_bgr == nullptr) {
    return false;
  }

  cv::Mat input;
  convertBGRToInputByType(*image_bgr, input);
  return extract(FramePackage{url, std::make_shared<cv::Mat>(input)}, feature);
}

cv::Mat FaceRecognition::estimateNorm(const std::vector<cv::Point2f> &landmarks,
                                      int imageSize) {
  assert(landmarks.size() == 5);
  assert(imageSize % 112 == 0 || imageSize % 128 == 0);

  float ratio;
  float diffX = 0.0;
  if (imageSize % 112 == 0) {
    ratio = static_cast<float>(imageSize) / 112.0f;
  } else {
    ratio = static_cast<float>(imageSize) / 128.0f;
    diffX = 8.0f * ratio;
  }

  // Assuming arcfaceDst is a predefined 5x2 matrix of facial landmarks for
  // normalization You need to define this matrix based on your specific use
  // case
  cv::Mat arcfaceDst =
      (cv::Mat_<float>(5, 2) << 38.2946f, 51.6963f, 73.5318f, 51.5014f,
       56.0252f, 71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f);

  cv::Mat dst = arcfaceDst * ratio;
  for (int i = 0; i < dst.rows; ++i) {
    dst.at<float>(i, 0) += diffX;
  }

  cv::Mat src(landmarks);
  cv::Mat tform = cv::estimateAffinePartial2D(src, dst);
  return tform;
}

cv::Mat FaceRecognition::normCrop(const cv::Mat &img,
                                  const std::vector<cv::Point2f> &landmarks,
                                  int imageSize) {
  cv::Mat M = estimateNorm(landmarks, imageSize);
  cv::Mat warped;
  cv::warpAffine(img, warped, M, cv::Size(imageSize, imageSize),
                 cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
  return warped;
}

void FaceRecognition::getFaceInput(cv::Mat const &input, cv::Mat &output,
                                   FrameInfo &frame, Points2f const &points,
                                   ColorType const &type) {
  // 关键点矫正
  std::vector<cv::Point2f> cvPoints;
  for (auto &p : points) {
    cvPoints.push_back(cv::Point2f{p.x, p.y});
  }
  if (type == ColorType::NV12) {
    cv::Mat temp;
    utils::NV12toRGB(input, temp);
    temp = normCrop(temp, cvPoints, 112);
    utils::RGB2NV12(temp, output);
    frame.shape = {temp.cols, temp.rows, output.channels()};
  } else {
    output = normCrop(input, cvPoints, 112);
    frame.shape = {output.cols, output.rows, output.channels()};
  }
  frame.inputShape = {output.cols, output.rows, output.channels()};
  frame.type = type;
  frame.data = reinterpret_cast<void **>(&output.data);
}

} // namespace server::face::core
