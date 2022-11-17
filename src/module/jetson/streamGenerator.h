/**
 * @file jetsonSourceModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_JETSON_SOURCE_H
#define __METAENGINE_JETSON_SOURCE_H

#include <any>
#include <memory>
#include <vector>

#include "common/common.hpp"
#include "common/config.hpp"
#include "logger/logger.hpp"
#include "messageBus.h"
#include "module.hpp"
#include "videoSource.h"
#include <opencv2/opencv.hpp>
namespace module {

void delBuffer(std::vector<std::any> &);
std::any getPtrBuffer(std::vector<std::any> &, FrameBuf *);
std::any getBuffer(std::vector<std::any> &, FrameBuf *);
FrameBuf makeFrameBuf(uchar3 *, int, int);

class StreamGenerator : public Module {
private:
  std::unique_ptr<videoSource> inputStream;

  uchar3 *frame = nullptr;

  int count = 0;

  videoOptions opt;

  CameraResult cameraResult;

  common::CameraConfig params;

public:
  StreamGenerator(Backend *ptr, const std::string &initName,
                     const std::string &initType,
                     const common::CameraConfig &_params);

  ~StreamGenerator() {}

  virtual void forward(std::vector<forwardMessage> message) override;

  void step() override;
};
} // namespace module
#endif // __METAENGINE_JETSON_SOURCE_H
