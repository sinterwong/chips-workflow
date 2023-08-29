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
  // 初始化视频解码
  virtual bool init() = 0;

  // 开启视频解码
  virtual bool start(std::string const &uri) = 0;

  // 关闭视频解码
  virtual bool stop() = 0;

  // 启动视频解码
  virtual bool run() = 0;

  // 解码状态检查
  virtual bool isRunning() = 0;

  // 基础信息获取
  virtual int getHeight() = 0;
  virtual int getWidth() = 0;
  virtual int getRate() = 0;

  // 获取解码图像
  virtual std::shared_ptr<cv::Mat> getcvImage() = 0;

  // 解码图像颜色格式
  virtual common::ColorType getType() const noexcept = 0;

  virtual ~VDecoder() noexcept {}
};
} // namespace video

#endif