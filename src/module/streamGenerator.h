/**
 * @file StreamGenerator.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.2
 * @date 2022-10-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_SUNRISE_DECODER_H
#define __METAENGINE_SUNRISE_DECODER_H

#include <any>
#include <memory>
#include <utility>
#include <vector>

#include "common/common.hpp"
#include "common/config.hpp"
#include "logger/logger.hpp"
#include "messageBus.h"
#include "module.hpp"

#if (TARGET_PLATFORM == 0)
#include "x3/videoDecoder.hpp"
#elif (TARGET_PLATFORM == 1)
#include "jetson/videoDecoder.hpp"
#endif

namespace module {

class StreamGenerator : public Module {

private:
  // 帧数索引
  int mmz_index = 0;
  // 流拉取状态
  int error;

  int ret = 0;

  CameraResult cameraResult;

  std::unique_ptr<utils::VideoDecoder> decoder;

  cv::Mat frame;

public:
  StreamGenerator(Backend *ptr, const std::string &initName,
                  const std::string &initType,
                  const common::CameraConfig &_params);

  ~StreamGenerator() {}

  virtual void beforeForward() override;

  virtual void forward(std::vector<forwardMessage> &message) override;

  void step() override;

  static void delBuffer(std::vector<std::any> &);

  // std::any getFrameInfo(std::vector<std::any> &, FrameBuf *);

  // std::any getMatBuffer(std::vector<std::any> &list, FrameBuf *buf);

  // std::any getPtrBuffer(std::vector<std::any> &list, FrameBuf *buf);
};

} // namespace module
#endif // __METAENGINE_JETSON_SOURCE_H
