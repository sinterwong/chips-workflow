/**
 * @file videoOutput.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-12
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "x3/videoOutput.hpp"
#include "logger/logger.hpp"
#include <sp_codec.h>
#include <sp_vio.h>

namespace module::utils {
bool videoOutput::open() {
  // chn 做成全局变量
  int ret = sp_start_encode(encoder, mOptions.videoIdx % 32, SP_ENCODER_H264,
                            mOptions.width, mOptions.height, 8000);
  if (ret != 0) {
    FLOWENGINE_LOGGER_ERROR("sp_open_encode failed {}!", ret);
    return false;
  }
  FLOWENGINE_LOGGER_INFO("sp_open_encode is successed!");
  mStreaming.store(true);
  return true;
}

bool videoOutput::close() noexcept {
  sp_stop_encode(encoder);
  return true;
}

bool videoOutput::render(void **image) { return true; }

} // namespace module::utils
