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

#include "video_common.hpp"
#include "xEncoder.hpp"

#include "vrecorder.hpp"

namespace video {

class VideoRecord : private VRecord {
public:
  explicit VideoRecord(videoOptions &&params_) : params(params_) {
    channel = ChannelsManager::getInstance().getChannel();
    if (channel < 0) {
      throw std::runtime_error("Channel usage overflow!");
    }
  }

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
  bool init() override;

  /**
   * @brief Destory the stream.
   *
   * @return true
   * @return false
   */
  void destory() noexcept override;

  /**
   * @brief Whether the stream is working.
   *
   * @return true
   * @return false
   */
  bool check() const noexcept override;

  /**
   * @brief Record the frame
   *
   * @return true
   * @return false
   */
  bool record(void *frame) override;

private:
  std::unique_ptr<XEncoder> stream = nullptr;
  videoOptions params;
  int channel;
};
} // namespace video