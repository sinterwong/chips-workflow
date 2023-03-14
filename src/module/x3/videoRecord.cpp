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
#include <filesystem>
#include <memory>

#include "module_utils.hpp"

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
  // if (!stream->close()) {
  //   return;
  // }
  // 将视频转成mp4格式
  std::string &location = params.resource.location;
  std::string renamed = location.substr(0, location.find(".")) + ".h264";
  std::cout << renamed << std::endl;
  std::filesystem::path oldPath(location);
  std::filesystem::path newPath(renamed);
  try {
    std::filesystem::rename(oldPath, newPath);
    FLOWENGINE_LOGGER_INFO("文件重命名成功");
  } catch (const std::filesystem::filesystem_error &e) {
    FLOWENGINE_LOGGER_ERROR("文件重命名失败：{}", e.what());
    return;
  }
  wrapH2642mp4(renamed, location);
}

bool VideoRecord::record(void *frame) {
  return stream->render(&frame);
  // return true;
}
} // namespace utils
} // namespace module