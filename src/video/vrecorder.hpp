/**
 * @file vrecorder.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-05-11
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "common/common.hpp"
namespace video {

class VRecord : private common::NonCopyable {

  virtual bool init() = 0;

  virtual void destory() noexcept = 0;

  virtual bool check() const noexcept = 0;

  virtual bool record(void *frame) = 0;
};
} // namespace video