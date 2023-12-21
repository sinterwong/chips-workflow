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
#include <opencv2/videoio.hpp>
#include <stdexcept>
#include <string>

#include "video_common.hpp"

#include "vrecorder.hpp"

namespace video {

class VideoRecord : private VRecord {
public:
  explicit VideoRecord(videoOptions &&params_) : params(params_) {}

  ~VideoRecord() {
    if (check()) {
      destory();
    }
    // 手动析构编码器，确保析构完成后再释放channel
    stream.reset();
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
  std::unique_ptr<cv::VideoWriter> stream = nullptr;
  videoOptions params;
};
} // namespace video