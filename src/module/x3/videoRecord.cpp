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

bool VideoRecord::check() const noexcept {
  return true;
}

void VideoRecord::destory() noexcept {
}

bool VideoRecord::record(void *frame) {
  return true;
}
} // namespace utils
} // namespace module