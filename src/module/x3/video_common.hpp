/**
 * @file video_common.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-12
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef _X3_VIDEO_COMMON_HPP_
#define _X3_VIDEO_COMMON_HPP_
#include "x3/uri.hpp"

namespace module::utils {
struct videoOptions {
  URI resource;
  int width;
  int height;
  int frameRate;
  int videoIdx;
};
} // namespace module::utils
#endif
