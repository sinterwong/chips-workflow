/**
 * @file xDecoder.hpp
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
#include "logger/logger.hpp"
#include "videoSource.hpp"
#include "x3/ffstream.hpp"

#include <mutex>
#include <sp_codec.h>
#include <sp_vio.h>

#include <chrono>
#include <utility>
using namespace std::chrono_literals;

namespace module::utils {

class XDecoder : public videoSource {

public:
  /**
   * Create a decoder from the provided video options.
   */
  static std::unique_ptr<XDecoder> create(videoOptions const &options) {
    std::unique_ptr<XDecoder> cam =
        std::unique_ptr<XDecoder>(new XDecoder(options));
    if (!cam) {
      return nullptr;
    }
    // initialize decoder (with fallback)
    if (!cam->open()) {
      FLOWENGINE_LOGGER_ERROR("XDecoder -- failed to create device!");
      return nullptr;
    }
    FLOWENGINE_LOGGER_ERROR("XDecoder -- successfully created device!");
    return cam;
  }

  /**
   * Destructor
   */
  ~XDecoder() { close(); };

  virtual bool capture(void **image,
                       size_t timeout = DEFAULT_TIMEOUT) override {

    int ret = sp_decoder_get_image(decoder, yuv_data);
    if (ret != 0) {
      FLOWENGINE_LOGGER_WARN("sp_decoder_get_image get next frame is failed!");
      return false;
    }
    // TODO 数据如何给出去? copy? 先写出来吧，这样是不安全的
    *image = reinterpret_cast<void *>(yuv_data);
    return true;
  }

  /**
   * Return the interface type
   */
  virtual inline size_t getType() const noexcept override { return Type; }

  /**
   * Unique type identifier of decoder class.
   */
  static const size_t Type = (1 << 0);

private:
  XDecoder(videoOptions const &options) : videoSource(options) {}
  const std::unordered_map<std::string, int> entypeMapping{
      std::pair<std::string, int>("h264", SP_ENCODER_H264),
      std::pair<std::string, int>("h265", SP_ENCODER_H265),
      std::pair<std::string, int>("mpeg", SP_ENCODER_MJPEG),
  };
  std::unique_ptr<FFStream> stream;
  void *decoder;
  char *yuv_data;
  void *raw_data;

private:
  virtual bool open() override {
    // 启动流
    stream = std::make_unique<FFStream>(mOptions.resource);
    if (!stream->openStream()) {
      FLOWENGINE_LOGGER_ERROR("can't open the stream {}!",
                              std::string(mOptions.resource));
      return false;
    }

    decoder = sp_init_decoder_module();
    int ret = sp_start_decode(decoder, "", mOptions.videoIdx,
                              entypeMapping.at(stream->getCodecType()),
                              stream->getWidth(), stream->getHeight());
    std::this_thread::sleep_for(2s);
    if (ret != 0) {
      FLOWENGINE_LOGGER_ERROR("sp_open_decoder failed {}!", ret);
      return false;
    }
    FLOWENGINE_LOGGER_INFO("sp_open_decoder is successed!");
    int yuv_size = FRAME_BUFFER_SIZE(mOptions.width, mOptions.height);
    yuv_data = reinterpret_cast<char *>(malloc(yuv_size * sizeof(char)));

    producter = std::make_unique<std::thread>(&XDecoder::producting, this);

    return true;
  };

  /**
   * Close the stream.
   * @see videoSource::Close()
   */
  virtual inline void close() noexcept override {
    sp_stop_decode(decoder);
    sp_release_decoder_module(decoder);
    free(yuv_data);
    if (stream->isRunning()) {
      stream->closeStream();
    }
    producter->join();
  }

  std::unique_ptr<std::thread> producter; // 生产者
  void producting() {
    while (stream->isRunning()) {
      int bufSize;
      bufSize = stream->getRawFrame(&raw_data);
      if (bufSize < 0) {
        bufSize = 0;
      }
      int ret =
          sp_decoder_set_image(decoder, reinterpret_cast<char *>(raw_data),
                               mOptions.videoIdx, bufSize, 0);
      if (ret != 0) {
        FLOWENGINE_LOGGER_WARN("sp_decoder_set_image is failed: {}", ret);
        std::this_thread::sleep_for(2s);
      }
    }
    FLOWENGINE_LOGGER_WARN("streaming is over: {}",
                           std::string(mOptions.resource));
  }
};
} // namespace module::utils
#endif