/**
 * @file videoRecord.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-11-21
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "common/common.hpp"
#include <any>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "videoOptions.h"
#include "videoOutput.h"

namespace video {

class VideoRecord : private common::NonCopyable {
public:
  explicit VideoRecord(videoOptions &&params_) : params(params_) {}

  ~VideoRecord() { destory(); }

  /**
   * @brief init the stream.
   *
   * @return true
   * @return false
   */
  bool init();

  /**
   * @brief Destory the stream.
   *
   * @return true
   * @return false
   */
  void destory() noexcept;

  /**
   * @brief Whether the stream is working.
   *
   * @return true
   * @return false
   */
  bool check() const noexcept;

  /**
   * @brief Record the frame
   *
   * @return true
   * @return false
   */
  bool record(void *frame);

private:
  std::unique_ptr<videoOutput> stream = nullptr;
  videoOptions params;
  int channel;
};
} // namespace video