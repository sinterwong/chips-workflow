/**
 * @file faceQuality.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-11-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __SERVER_FACE_CORE_FACE_QUALITY_HPP_
#define __SERVER_FACE_CORE_FACE_QUALITY_HPP_

#include "algoManager.hpp"
#include <mutex>
namespace server::face::core {

class FaceQuality {
public:
  static FaceQuality &getInstance() {
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] { instance = new FaceQuality(); });
    return *instance;
  }

  FaceQuality(FaceQuality const &) = delete;
  FaceQuality &operator=(FaceQuality const &) = delete;

public:
  /**
   * @brief
   *
   * @param framePackage
   * @param quality
   * 遮挡类别：胡子(0), 正常(1), 眼镜(2), 口罩(3), 墨镜(4), 遮挡(5）
   * 检测判定：1: 图小, 2: 长宽比
   * 截图判定：3: 锐度和亮度
   * 关键点判定：4: 角度大
   * 分类判定：5: 胡子, 0: 正常, 6: 眼镜, 7: 口罩, 8: 墨镜, 9: 遮挡
   * @return true
   * @return false
   */

  bool infer(std::string const &url, int &quality);

  bool infer(FramePackage const &framePackage, int &quality);

private:
  FaceQuality() {}
  ~FaceQuality() {
    delete instance;
    instance = nullptr;
  }
  static FaceQuality *instance;

private:
  bool getFaceInput(cv::Mat const &input, cv::Mat &output, FrameInfo &frame,
                    BBox const &bbox, ColorType const &type);

  void points5angle(Points2f const &points, float &pitch, float &yaw,
                    float &roll);
};
} // namespace server::face::core

#endif // __SERVER_FACE_CORE_FACE_QUALITY_HPP_