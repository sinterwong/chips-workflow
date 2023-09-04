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

using TIMEPOINT = std::chrono::time_point<std::chrono::high_resolution_clock>;

private:
  // CameraResult cameraResult;
  std::unique_ptr<StreamBase> config;

  std::unique_ptr<VideoDecode> decoder;

  void messageListener(); // 监听外部消息

  // 开始和结束执行程序的时间
  TIMEPOINT startTime;
  TIMEPOINT endTime;

public:
  StreamModule(backend_ptr ptr, std::string const &, MessageType const &,
               ModuleConfig &) noexcept(false);

  ~StreamModule() { ptr->pools->unregistered(name); }

  virtual void go() override;

  virtual void beforeForward() override;

  virtual void afterForward() override;

  virtual void forward(std::vector<forwardMessage> &message) override{};

  void startup();

  void step() override;
};

} // namespace module
#endif // __METAENGINE_JETSON_SOURCE_H
