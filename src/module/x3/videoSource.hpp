/**
 * @file videoSource.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-04
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __X3_VIDEO_SOURCE_H_
#define __X3_VIDEO_SOURCE_H_

#include "common/common.hpp"
#include "x3/uri.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>

using common::ColorType;

namespace module::utils {

struct videoOptions {
  URI resource;
  int width;
  int height;
  int frameRate;
  int videoIdx;
};

class videoSource {
public:
  static std::unique_ptr<videoSource> create(videoOptions const &options);

  virtual ~videoSource() {}

  virtual bool capture(void **image, uint64_t timeout = DEFAULT_TIMEOUT) = 0;

  virtual bool open() = 0;

  virtual void close() noexcept = 0;

  inline bool isStreaming() const noexcept { return mStreaming; }

  inline size_t getWidth() const noexcept { return mOptions.width; }

  inline size_t getHeight() const noexcept { return mOptions.height; }

  inline size_t getFrameRate() const noexcept { return mOptions.frameRate; }

  uint64_t getLastTimestamp() const noexcept { return mLastTimestamp; }

  inline ColorType getRawFormat() const noexcept { return mRawFormat; }

  inline const URI &getResource() const noexcept { return mOptions.resource; }

  inline const videoOptions &getOptions() const noexcept { return mOptions; }

  virtual inline size_t getType() const noexcept { return 0; }

  inline bool isType(size_t type) const noexcept { return (type == getType()); }

  template <typename T> bool isType() const noexcept { return isType(T::Type); }

  inline const std::string typeTostr() const { return typeTostr(getType()); }

  static const std::string typeTostr(size_t type);

  static constexpr size_t DEFAULT_TIMEOUT = 1000;

protected:
  ColorType mRawFormat;
  std::atomic<bool> mStreaming;
  videoOptions mOptions;
  size_t mLastTimestamp;

  videoSource(videoOptions const &options) : mOptions(options) {
    mStreaming = false;
    mLastTimestamp = 0;
    mRawFormat = ColorType::None;
  }
};
} // namespace module::utils
#endif
