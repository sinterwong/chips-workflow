/**
 * @file videoDecode.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-12-20
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "videoDecode.hpp"

using namespace std::chrono_literals;

namespace video {

bool VideoDecode::init() { return true; }

bool VideoDecode::start(const std::string &url) { return true; }

bool VideoDecode::stop() { return true; }

bool VideoDecode::run() { return true; }

std::shared_ptr<cv::Mat> VideoDecode::getcvImage() { return nullptr; }

void VideoDecode::consumeFrame() {}

} // namespace video