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
#include "videoDecode.hpp"

using common::ModuleConfig;
using common::StreamBase;
using namespace video;

namespace module {

class StreamModule : public Module {

private:
  // CameraResult cameraResult;
  std::unique_ptr<StreamBase> config;

  std::unique_ptr<VideoDecode> decoder;

public:
  StreamModule(backend_ptr ptr, std::string const &, MessageType const &,
               ModuleConfig &) noexcept(false);

  ~StreamModule() { ptr->pools->unregistered(name); }

  virtual void beforeForward() override;

  virtual void forward(std::vector<forwardMessage> &message) override{};

  void startup();

  void step() override;
};

} // namespace module
#endif // __METAENGINE_JETSON_SOURCE_H
