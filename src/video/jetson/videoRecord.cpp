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
#include "video_utils.hpp"
#include <algorithm>
#include <cassert>
#include <memory>

namespace video {

bool VideoRecord::init() {
  stream = std::unique_ptr<videoOutput>(videoOutput::Create(std::move(params)));
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
  stream->Render(reinterpret_cast<uchar3 *>(frame), params.width,
                 params.height);
  char str[256];
  sprintf(str, " (%dx%d)", params.width, params.height);
  // update status bar
  stream->SetStatus(str);
  return true;
}
} // namespace video
