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

#include "common/joining_thread.h"
#include "ffstream.hpp"
#include "logger/logger.hpp"
#include "videoSource.hpp"
#include "video_common.hpp"

#include <atomic>
#include <memory>
#include <sp_codec.h>
#include <sp_vio.h>

#include <chrono>
#include <thread>
#include <utility>
using namespace std::chrono_literals;
namespace video {

class XDecoder : public videoSource {

public:
  /**
   * Create a decoder from the provided video options.
   */
  static std::unique_ptr<XDecoder> Create(videoOptions const &options) {
    std::unique_ptr<XDecoder> cam =
        std::unique_ptr<XDecoder>(new XDecoder(options));
    if (!cam) {
      return nullptr;
    }
    // initialize decoder (with fallback)
    if (!cam->Init()) {
      FLOWENGINE_LOGGER_ERROR("XDecoder -- failed to create device!");
      return nullptr;
    }
    FLOWENGINE_LOGGER_INFO("XDecoder -- successfully created device!");
    return cam;
  }

  /**
   * Create a decoder without options.
   */
  static std::unique_ptr<XDecoder> Create() {
    std::unique_ptr<XDecoder> cam = std::unique_ptr<XDecoder>(new XDecoder());
    if (!cam) {
      return nullptr;
    }
    // initialize decoder (with fallback)
    if (!cam->Init()) {
      FLOWENGINE_LOGGER_ERROR("XDecoder -- failed to create device!");
      return nullptr;
    }
    FLOWENGINE_LOGGER_INFO("XDecoder -- successfully created device!");
    return cam;
  }

  /**
   * Destructor
   */
  ~XDecoder() {
    if (mStreaming.load()) {
      Close();
    }
    sp_release_decoder_module(decoder);
  };

  virtual bool Open() override {
    if (!mOptions) {
      // 没有初始化过参数配置
      FLOWENGINE_LOGGER_CRITICAL(
          "Please offer the options before using Open().");
      return false;
    }
    std::lock_guard<std::mutex> lk(resourceMutex);
    // 启动流
    stream = std::make_unique<FFStream>(mOptions->resource);
    if (!stream->openStream()) {
      FLOWENGINE_LOGGER_ERROR("can't open the stream {}!",
                              std::string(mOptions->resource));
      return false;
    }
    mOptions->width = stream->getWidth();
    mOptions->height = stream->getHeight();
    mOptions->frameRate = stream->getRate();
    FLOWENGINE_LOGGER_INFO("{} video is opened!", mOptions->resource.string);

    int ret = sp_start_decode(decoder, "", mOptions->videoIdx,
                              entypeMapping.at(stream->getCodecType()),
                              stream->getWidth(), stream->getHeight());
    if (ret != 0) {
      FLOWENGINE_LOGGER_ERROR("sp_open_decoder failed {}!", ret);
      return false;
    }
    FLOWENGINE_LOGGER_INFO("sp_open_decoder is successed!");
    int yuv_size = FRAME_BUFFER_SIZE(mOptions->width, mOptions->height);
    // yuv_data = reinterpret_cast<char *>(malloc(yuv_size * sizeof(char)));
    yuv_data = new char[yuv_size];
    // raw_data = malloc(mOptions->width * mOptions->height * 3 * sizeof(char));
    mStreaming.store(true);
    producter = std::make_unique<joining_thread>([this]() {
      isClosed.store(false);
      // 优化成一个条件变量
      while (!mTerminate.load() && stream && stream->isRunning()) {
        int bufSize = stream->getDataFrame(&raw_data);
        if (bufSize < 0) {
          FLOWENGINE_LOGGER_ERROR("stream is over!");
          break;
        } else if (bufSize == 0) {
          std::this_thread::sleep_for(10ms);
          FLOWENGINE_LOGGER_WARN("stream is empty!");
          continue;
        }
        int ret =
            sp_decoder_set_image(decoder, reinterpret_cast<char *>(raw_data),
                                 mOptions->videoIdx, bufSize, 0);
        if (ret != 0) {
          FLOWENGINE_LOGGER_WARN("sp_decoder_set_image is failed: {}", ret);
          std::this_thread::sleep_for(20ms);
          continue;
        }
      }
      std::lock_guard<std::mutex> lock(closeMutex);
      isClosed.store(true);
      cv.notify_one(); // 通知析构函数线程已经结束

      // 到此处如果资源没有释放，说明是流主动关闭，需要手动释放资源
      if (mStreaming.load()) {
        releaseDecoderResource();
      }
    });
    return true;
  }

  virtual bool Open(videoOptions const &options) override {
    // 这里初始化一下参数即可
    mOptions = std::make_unique<videoOptions>(options);
    return Open();
  }

  /**
   * Close the stream.
   * @see videoSource::Close()
   */
  virtual inline void Close() noexcept override {
    // 外界关闭，需要和生产者线程同步

    std::lock_guard<std::mutex> lk(resourceMutex);
    // 通知生产者线程结束
    mTerminate.store(true);
    {
      // 等待生产者线程结束
      std::unique_lock<std::mutex> closeLock(closeMutex);
      cv.wait(closeLock, [this]() { return isClosed.load(); });
      mTerminate.store(false); // 重置
    }
    if (mStreaming.load()) {
      releaseDecoderResource();
    }
  }

  virtual inline size_t GetWidth() const noexcept override {
    if (!stream) {
      FLOWENGINE_LOGGER_ERROR("Stream object is null!");
      return 0; // Or another default value.
    }
    return stream->getWidth();
  }

  virtual inline size_t GetHeight() const noexcept override {
    if (!stream) {
      FLOWENGINE_LOGGER_ERROR("Stream object is null!");
      return 0; // Or another default value.
    }
    return stream->getHeight();
  }

  virtual inline size_t GetFrameRate() const noexcept override {
    if (!stream) {
      FLOWENGINE_LOGGER_ERROR("Stream object is null!");
      return 0; // Or another default value.
    }
    return stream->getRate();
  }

  virtual bool Capture(void **image,
                       size_t timeout = DEFAULT_TIMEOUT) override {
    std::lock_guard<std::mutex> lk(resourceMutex);
    if (!mStreaming.load()) {
      return false;
    }

    // TODO 如果数次Capture获取不到数据应该有点动作
    int ret = sp_decoder_get_image(decoder, yuv_data);
    if (ret != 0) {
      FLOWENGINE_LOGGER_WARN("sp_decoder_get_image get next frame is failed!");
      std::this_thread::sleep_for(10ms);
      return false;
    }
    *image = reinterpret_cast<void *>(yuv_data);
    return true;
  }

  /**
   * Return the interface type
   */
  virtual inline size_t GetType() const noexcept override { return Type; }

  /**
   * Unique type identifier of decoder class.
   */
  static const size_t Type = (1 << 0);

private:
  XDecoder(videoOptions const &options) : videoSource(options) {}

  XDecoder() : videoSource() {}

  const std::unordered_map<std::string, int> entypeMapping{
      std::pair<std::string, int>("h264", SP_ENCODER_H264),
      std::pair<std::string, int>("h265", SP_ENCODER_H265),
      std::pair<std::string, int>("mpeg", SP_ENCODER_MJPEG),
  };
  std::unique_ptr<FFStream> stream;
  void *decoder;
  char *yuv_data;
  void *raw_data;
  std::mutex resourceMutex;

  // 用于结束生产者线程
  std::mutex closeMutex;
  std::atomic<bool> mTerminate{false}; // 主动关闭时需要使用
  std::atomic<bool> isClosed{true};
  std::condition_variable cv;

  virtual bool Init() override {
    decoder = sp_init_decoder_module();
    return true;
  }
  std::unique_ptr<joining_thread> producter;

  void releaseDecoderResource() {
    // 停止解码
    sp_stop_decode(decoder);

    // 结束流
    if (stream && stream->isRunning()) {
      stream->closeStream();
      stream.reset();
    }
    // 释放资源
    if (yuv_data) {
      delete[] yuv_data;
      yuv_data = nullptr;
    }
    mStreaming.store(false);
  }
};
} // namespace video
#endif