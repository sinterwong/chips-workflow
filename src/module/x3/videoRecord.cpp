/**
 * @file videoRecord.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "videoRecord.hpp"
#include "logger/logger.hpp"
#include <algorithm>
#include <cassert>
#include <memory>

namespace module {
namespace utils {

bool VideoRecord::init() {
  params.videoIdx = channel;
  FLOWENGINE_LOGGER_CRITICAL("Recording video in channel {}", channel);
  stream = XEncoder::create(params);
  stream->init();
  if (!stream) {
    return false;
  }
  return true;
}

bool VideoRecord::check() const noexcept {
  return stream && stream->isStreaming();
  // return true;
}

void VideoRecord::destory() noexcept {
  // 不做处理, XEncoder 在析构的时候会自行释放
}

bool VideoRecord::record(void *frame) {
  return stream->render(&frame);
  // return true;
}
} // namespace utils
} // namespace module