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

#if (TARGET_PLATFORM == 0)
#include "x3/video_common.hpp"
#include "x3/xEncoder.hpp"
using namespace module::utils;
#elif (TARGET_PLATFORM == 1)
#include "videoOptions.h"
#include "videoOutput.h"
#elif (TARGET_PLATFORM == 2)
#endif

namespace module {
namespace utils {

class VideoRecord : private common::NonCopyable {
public:
  explicit VideoRecord(videoOptions &&params_) : params(params_) {
#if (TARGET_PLATFORM == 0)
    channel = ChannelsManager::getInstance().getChannel();
    if (channel < 0) {
      throw std::runtime_error("Channel usage overflow!");
    }
#endif
  }

  ~VideoRecord() {
    destory();
#if (TARGET_PLATFORM == 0)
    ChannelsManager::getInstance().setChannel(channel);
#endif
  }

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
#if (TARGET_PLATFORM == 0)
  std::unique_ptr<XEncoder> stream = nullptr;
#elif (TARGET_PLATFORM == 1)
  std::unique_ptr<videoOutput> stream = nullptr;
#endif
  videoOptions params;
  int channel;
};
} // namespace utils
} // namespace module