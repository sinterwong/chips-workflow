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

#include <any>
#include <iostream>
#include <memory>
#include <string>

namespace module {
namespace utils {
struct VideoParams {
  std::string uri;
  int height;
  int width;
  int rate;
};

class VideoRecord {
public:
  explicit VideoRecord(VideoParams &&params_)
      : params(params_) {
    // stream = std::unique_ptr<videoOutput>(videoOutput::Create(std::move(opt)));
  }

  VideoRecord(VideoRecord const &other) = delete;
  VideoRecord(VideoRecord &&other) = delete;
  VideoRecord &operator=(VideoRecord const &other) = delete;
  VideoRecord &operator=(VideoRecord &&other) = delete;

  ~VideoRecord() {
    destory();
  }

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
  // std::unique_ptr<videoOutput> stream = nullptr;
  VideoParams params;
};
} // namespace utils
} // namespace module