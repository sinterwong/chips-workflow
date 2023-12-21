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
#include <experimental/filesystem>

using namespace std::experimental;

namespace video {

bool VideoRecord::init() { return false; }

bool VideoRecord::check() const noexcept { return false; }

void VideoRecord::destory() noexcept {}

bool VideoRecord::record(void *frame) { return false; }
} // namespace video
