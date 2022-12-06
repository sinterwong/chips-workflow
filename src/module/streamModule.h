/**
 * @file streamModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-12-06
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
#include "x3/videoManager.hpp"
using namespace module::utils;
#elif (TARGET_PLATFORM == 1)
#include "jetson/videoManager.hpp"
#endif

namespace module {

class StreamModule : public Module {

private:
  CameraResult cameraResult;

  std::unique_ptr<VideoManager> vm;

public:
  StreamModule(Backend *ptr, const std::string &initName,
                  const std::string &initType,
                  const common::CameraConfig &_params);

  ~StreamModule() {}

  virtual void beforeForward() override;

  virtual void forward(std::vector<forwardMessage> &message) override;

  void step() override;

  static void delBuffer(std::vector<std::any> &);

  std::any getFrameInfo(std::vector<std::any> &, FrameBuf *);

  std::any getMatBuffer(std::vector<std::any> &list, FrameBuf *buf);

  std::any getPtrBuffer(std::vector<std::any> &list, FrameBuf *buf);
};

} // namespace module
#endif // __METAENGINE_JETSON_SOURCE_H
