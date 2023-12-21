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

bool VideoRecord::init() { return true; }

bool VideoRecord::check() const noexcept { return true; }

void VideoRecord::destory() noexcept {}

bool VideoRecord::record(void *frame) { return true; }
} // namespace video
