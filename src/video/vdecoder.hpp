/**
 * @file videoManager.hpp
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
  virtual bool init() = 0;
  virtual bool run() = 0;
  virtual bool isRunning() = 0;
  virtual int getHeight() = 0;
  virtual int getWidth() = 0;
  virtual int getRate() = 0;
  virtual std::shared_ptr<cv::Mat> getcvImage() = 0;
  virtual common::ColorType getType() const noexcept = 0;
  virtual ~VDecoder() noexcept {}
};
} // namespace video

#endif