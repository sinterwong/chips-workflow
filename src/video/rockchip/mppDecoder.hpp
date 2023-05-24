/**
 * @file MPPDecoder.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-05
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __DECODER_FOR_X3_H_
#define __DECODER_FOR_X3_H_

#include "ffstream.hpp"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include "video_common.hpp"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <shared_mutex>

#include <chrono>
#include <thread>
#include <utility>
using namespace std::chrono_literals;
namespace video {

class MPPDecoder {

public:
  MPPDecoder(videoOptions const &options) : mOptions(options) {}

  ~MPPDecoder(){};

  bool init() { return true; }

  bool open() {
    // // 启动流
    // stream = std::make_unique<utils::FFStream>(mOptions.resource);
    // if (!stream->openStream()) {
    //   FLOWENGINE_LOGGER_ERROR("can't open the stream {}!",
    //                           std::string(mOptions.resource));
    //   return false;
    // }
    // mOptions.width = stream->getWidth();
    // mOptions.height = stream->getHeight();
    // mOptions.frameRate = stream->getRate();
    // FLOWENGINE_LOGGER_INFO("{} video is opened!", mOptions.resource.string);
    return true;
  };

  /**
   * Close the stream.
   * @see videoSource::Close()
   */
  inline void close() noexcept {}

  inline size_t getWidth() const noexcept { return stream->getWidth(); }

  inline size_t getHeight() const noexcept { return stream->getHeight(); }

  inline size_t getFrameRate() const noexcept { return stream->getRate(); }

  bool capture(void **image, size_t timeout = 1000) { return true; }

  /**
   * Return the interface type
   */
  inline size_t getType() const noexcept { return Type; }

  /**
   * Unique type identifier of decoder class.
   */
  static const size_t Type = (1 << 0);

private:
  videoOptions mOptions;
  std::unique_ptr<utils::FFStream> stream;
  std::unique_ptr<joining_thread> producter;
};
} // namespace video
#endif