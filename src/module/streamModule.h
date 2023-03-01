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

#ifndef __FLOWENGINE_STREAM_MODULE_H_
#define __FLOWENGINE_STREAM_MODULE_H_

#include <any>
#include <memory>
#include <utility>
#include <vector>

#include "common/common.hpp"

#include "logger/logger.hpp"
#include "messageBus.h"
#include "module.hpp"

#include "videoManager.hpp"

using namespace module::utils;
using common::StreamBase;

namespace module {

class StreamModule : public Module {

private:
  // CameraResult cameraResult;
  StreamBase config;

  std::unique_ptr<VideoManager> vm;

public:
  StreamModule(backend_ptr ptr, std::string const &, std::string const &,
               StreamBase const &);

  ~StreamModule() {}

  virtual void beforeForward() override;

  virtual void forward(std::vector<forwardMessage> &message) override;

  void step() override;

  static void delBuffer(std::vector<std::any> &);

  std::any getMatBuffer(std::vector<std::any> &list, FrameBuf *buf);

  std::any getPtrBuffer(std::vector<std::any> &list, FrameBuf *buf);
};

} // namespace module
#endif // __METAENGINE_JETSON_SOURCE_H
