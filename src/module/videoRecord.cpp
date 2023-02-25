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
#if (TARGET_PLATFORM == 0)
  params.videoIdx = channel;
  FLOWENGINE_LOGGER_CRITICAL("Recording video in channel {}", channel);
  stream = XEncoder::Create(params);
  stream->Init();
#elif (TARGET_PLATFORM == 1)
  stream = std::unique_ptr<videoOutput>(videoOutput::Create(std::move(params)));
#endif
  if (!stream) {
    return false;
  }
  return true;
}

bool VideoRecord::check() const noexcept {
  return stream && stream->IsStreaming();
}

void VideoRecord::destory() noexcept {
  if (check()) {
    stream->Close();
  }
  stream = nullptr;
}

bool VideoRecord::record(void *frame) {
#if (TARGET_PLATFORM == 0)
  return stream->Render(&frame);
#elif (TARGET_PLATFORM == 1)
  stream->Render(reinterpret_cast<uchar3 *>(frame), params.width,
                 params.height);
  char str[256];
  sprintf(str, " (%dx%d)", params.width, params.height);
  // update status bar
  stream->SetStatus(str);
  return true;
#endif
}
} // namespace utils
} // namespace module
