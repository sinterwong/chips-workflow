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
#include "x3/video_common.hpp"
#include <atomic>
#include <memory>
#include <sp_codec.h>

namespace module::utils {

class videoOutput {
public:
  static std::unique_ptr<videoOutput> create(videoOptions const &options);

  virtual ~videoOutput() {}

  /**
   * @brief 初始化编码组建
   *
   * @return true
   * @return false
   */
  inline bool init() {
    encoder = sp_init_encoder_module();
    return true;
  }

  /**
   * @brief 开始编码
   *
   * @return true
   * @return false
   */
  bool open();

  bool close() noexcept;

  /**
   * @brief
   *
   * @param image nv12 image
   * @param timeout
   * @return true
   * @return false
   */
  bool render(void **image);

  inline bool isStreaming() const noexcept { return mStreaming; }

  inline size_t getWidth() const noexcept { return mOptions.width; }

  inline size_t getHeight() const noexcept { return mOptions.height; }

  inline size_t getFrameRate() const noexcept { return mOptions.frameRate; }

  inline const URI &getResource() const noexcept { return mOptions.resource; }

  inline const videoOptions &getOptions() const noexcept { return mOptions; }

private:
  std::atomic<bool> mStreaming;
  videoOptions mOptions;
  void *encoder;

  videoOutput(videoOptions const &options) : mOptions(options) {
    mStreaming = false;
  }
};
} // namespace module::utils
#endif
