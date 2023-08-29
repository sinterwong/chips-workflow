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
#include "logger/logger.hpp"
#include "video_common.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>

using common::ColorType;

namespace video {

class videoSource {
public:
  static std::unique_ptr<videoSource> Create(videoOptions const &options);

  static std::unique_ptr<videoSource> Create();

  virtual ~videoSource() {}

  virtual bool Capture(void **image, uint64_t timeout = DEFAULT_TIMEOUT) = 0;

  virtual bool Init() = 0;

  virtual bool Open() = 0;

  virtual bool Open(videoOptions const &options) = 0;

  virtual void Close() noexcept = 0;

  inline bool IsStreaming() const noexcept { return mStreaming.load(); }

  virtual inline size_t GetWidth() const noexcept { return mOptions->width; }

  virtual inline size_t GetHeight() const noexcept { return mOptions->height; }

  virtual inline size_t GetFrameRate() const noexcept {
    return mOptions->frameRate;
  }

  uint64_t GetLastTimestamp() const noexcept { return mLastTimestamp; }

  inline ColorType GetRawFormat() const noexcept { return mRawFormat; }

  inline const URI &GetResource() const noexcept { return mOptions->resource; }

  inline const videoOptions &GetOptions() const noexcept { return *mOptions; }

  virtual inline size_t GetType() const noexcept { return 0; }

  inline bool IsType(size_t type) const noexcept { return (type == GetType()); }

  template <typename T> bool IsType() const noexcept { return isType(T::Type); }

  inline const std::string typeTostr() const { return typeTostr(GetType()); }

  static const std::string typeTostr(size_t type);

  static constexpr size_t DEFAULT_TIMEOUT = 1000;

protected:
  ColorType mRawFormat;
  std::atomic<bool> mStreaming;
  std::unique_ptr<videoOptions> mOptions;
  size_t mLastTimestamp;

  videoSource() {
    mStreaming.store(false);
    mLastTimestamp = 0;
    mRawFormat = ColorType::NV12;
    mOptions = nullptr;
  }

  videoSource(videoOptions const &options)
      : mOptions(std::make_unique<videoOptions>(options)) {
    mStreaming.store(false);
    mLastTimestamp = 0;
    mRawFormat = ColorType::NV12;
  }
};
} // namespace video
#endif
