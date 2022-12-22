/**
 * @file common.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-04-23
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <array>
#include <cstdint>
#include <getopt.h>
#include <iostream>
#include <string>
#include <vector>

#ifndef _FLOWENGINE_COMMON_COMMON_HPP_
#define _FLOWENGINE_COMMON_COMMON_HPP_

namespace common {
/**
 * @brief 颜色类型
 *
 */
enum class ColorType { RGB888 = 0, BGR888, NV12 };

/**
 * @brief 帧信息
 *
 */
struct FrameInfo {
  std::array<int, 3> shape;
  ColorType type;
  void **data;
};

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};

} // namespace common
#endif
