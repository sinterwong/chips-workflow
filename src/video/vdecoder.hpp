/**
 * @file vdecoder.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-12-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __FLOWENGINE_VIDEO_DECODER_INTERFACE_H_
#define __FLOWENGINE_VIDEO_DECODER_INTERFACE_H_

#include "common/common.hpp"
#include <memory>
#include <opencv2/core/mat.hpp>
namespace video {

class VDecoder : private common::NonCopyable {
public:
  /**
   * @brief
   * 初始化视频解码器（仅用于某些解码器和视频流没有封装在一起的情况，需要单独在这里初始化解码器）
   * 这个函数应该只和解码资源有关，无需视频流信息
   * @return true
   * @return false
   */
  virtual bool init() = 0;

  /**
   * @brief 启动视频解码任务
   * 注意：某些视频类型如csi需要提供分辨率信息，后续参数增多时可以考虑使用结构体传参
   * @param uri
   * @return true
   * @return false
   */
  virtual bool start(std::string const &uri, int width, int height) = 0;

  /**
   * @brief 关闭当前解码器的解码任务
   * 
   * @return true 
   * @return false 
   */
  virtual bool stop() = 0;

  /**
   * @brief 解码状态检查
   * 
   * @return true 
   * @return false 
   */
  virtual bool isRunning() = 0;

  /**
   * @brief 获取当前解码器解码的视频流的基本流信息
   * 
   * @return int 
   */
  virtual int getHeight() = 0;
  virtual int getWidth() = 0;
  virtual int getRate() = 0;

  /**
   * @brief 获取当前解码器解码的视频流的一帧图像
   * 
   * @return std::shared_ptr<cv::Mat> 
   */
  virtual std::shared_ptr<cv::Mat> getcvImage() = 0;

  /**
   * @brief 获取该平台的视频流颜色类型
   * 
   * @return common::ColorType 
   */
  virtual common::ColorType getType() const noexcept = 0;

  virtual ~VDecoder() noexcept {}
};
} // namespace video

#endif