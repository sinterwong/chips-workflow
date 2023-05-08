/**
 * @file videoOutput.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-12
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __X3_VIDEO_OUTPUT_H_
#define __X3_VIDEO_OUTPUT_H_

#include "common/common.hpp"
#include "video_common.hpp"
#include <atomic>
#include <fstream>
#include <memory>
#include <sp_codec.h>
#include <sp_vio.h>

namespace video {

class XEncoder {
public:
  static std::unique_ptr<XEncoder> Create(videoOptions const &options);

  virtual ~XEncoder();

  /**
   * @brief 初始化编码组建
   *
   * @return true
   * @return false
   */
  bool Init();

  /**
   * @brief 开始编码
   *
   * @return true
   * @return false
   */
  bool Open() noexcept;

  /**
   * @brief 关闭编码
   *
   * @return true
   * @return false
   */
  bool Close() noexcept;

  /**
   * @brief
   *
   * @param image nv12 image
   * @param timeout
   * @return true
   * @return false
   */
  bool Render(void **image);

  inline bool IsStreaming() const noexcept { return mStreaming; }

  inline size_t GetWidth() const noexcept { return mOptions.width; }

  inline size_t GetHeight() const noexcept { return mOptions.height; }

  inline size_t GetFrameRate() const noexcept { return mOptions.frameRate; }

  inline const URI &GetResource() const noexcept { return mOptions.resource; }

  inline const videoOptions &GetOptions() const noexcept { return mOptions; }

private:
  std::atomic<bool> mStreaming;
  videoOptions mOptions;
  void *encoder = nullptr;
  int frame_size = 0;
  char *stream_buffer = nullptr;
  std::ofstream outStream;

  XEncoder(videoOptions const &options) : mOptions(options) {
    mStreaming = false;
    frame_size = FRAME_BUFFER_SIZE(mOptions.height, mOptions.width);
  }
};
} // namespace video
#endif
