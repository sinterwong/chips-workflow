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
#include "module_utils.hpp"
#include <algorithm>
#include <cassert>
#include <experimental/filesystem>
#include <memory>

using namespace std::experimental;

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
#if (TARGET_PLATFORM == 0)
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
#endif
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
