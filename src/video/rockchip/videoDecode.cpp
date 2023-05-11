/**
 * @file videoDecode.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-11
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "videoDecode.hpp"

using namespace std::chrono_literals;

namespace video {

bool VideoDecode::init() { return true; }

void VideoDecode::consumeFrame() {}

bool VideoDecode::run() { return true; }

std::shared_ptr<cv::Mat> VideoDecode::getcvImage() { return nullptr; }
} // namespace video