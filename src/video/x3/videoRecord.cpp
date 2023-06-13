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
#include <experimental/filesystem>
#include <memory>

using namespace std::experimental;

namespace video {

bool VideoRecord::init() {
  params.videoIdx = channel;
  FLOWENGINE_LOGGER_DEBUG("Recording video in channel {}", channel);
  stream = XEncoder::Create(params);
  stream->Init();
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
  // 将视频转成mp4格式
  std::string &location = params.resource.location;
  std::string renamed = location.substr(0, location.find(".")) + ".h264";
  filesystem::path oldPath(location);
  filesystem::path newPath(renamed);
  try {
    filesystem::rename(oldPath, newPath);
    FLOWENGINE_LOGGER_INFO("文件重命名成功");
  } catch (const filesystem::filesystem_error &e) {
    FLOWENGINE_LOGGER_ERROR("文件重命名失败：{}", e.what());
    return;
  }
  wrapH2642mp4(renamed, location);
}

bool VideoRecord::record(void *frame) { return stream->Render(&frame); }
} // namespace video
