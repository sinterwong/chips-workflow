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
#include "x3/video_common.hpp"
#include "x3/xEncoder.hpp"
#include <any>
#include <iostream>
#include <memory>
#include <string>

namespace module {
namespace utils {

class VideoRecord : private common::NonCopyable {
public:
  explicit VideoRecord(videoOptions &&params_)
      : params(params_), channel(ChannelsManager::getInstance().getChannel()) {}

  ~VideoRecord() {
    destory();
    ChannelsManager::getInstance().setChannel(channel);
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
  std::unique_ptr<XEncoder> stream = nullptr;
  videoOptions params;
  int channel;
};
} // namespace utils
} // namespace module