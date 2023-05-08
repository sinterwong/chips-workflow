/**
 * @file xCamera.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-01-05
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef __CAMERA_FOR_X3_H_
#define __CAMERA_FOR_X3_H_

#include "logger/logger.hpp"
#include "videoSource.hpp"
#include <chrono>
#include <memory>

#include <sp_vio.h>
#include <thread>

using namespace std::chrono_literals;

namespace video {

class XCamera : public videoSource {
public:
  /**
   * Create a decoder from the provided video options.
   */
  static std::unique_ptr<XCamera> Create(videoOptions const &options) {
    std::unique_ptr<XCamera> cam =
        std::unique_ptr<XCamera>(new XCamera(options));
    if (!cam) {
      return nullptr;
    }
    // initialize camera (with fallback)
    if (!cam->Init()) {
      FLOWENGINE_LOGGER_ERROR("XCamera -- failed to create device!");
      return nullptr;
    }
    FLOWENGINE_LOGGER_INFO("XCamera -- successfully created device!");
    return cam;
  }

  /**
   * Destructor
   */
  ~XCamera() {
    if (mStreaming.load()) {
      Close();
    }
    sp_release_vio_module(camera);
  };

  virtual bool Capture(void **image,
                       size_t timeout = DEFAULT_TIMEOUT) override {

    if (!mStreaming.load()) // TODO
      if (!Open())
        return false;

    int ret = sp_vio_get_yuv(camera, yuv_data, mOptions.width, mOptions.height,
                             DEFAULT_TIMEOUT);
    if (ret != 0) {
      FLOWENGINE_LOGGER_WARN("sp_vio_get_yuv get next frame is failed!");
      return false;
    }
    // TODO 数据如何给出去? copy? 先写出来吧，这样是不安全的
    *image = reinterpret_cast<void *>(yuv_data);
    return true;
  }

  /**
   * Open the stream.
   * @see videoSource::Open()
   */
  virtual bool Open() override {
    int ret = sp_open_camera(camera, 0, mOptions.videoIdx, &mOptions.width,
                             &mOptions.height);
    std::this_thread::sleep_for(2s);
    if (ret != 0) {
      FLOWENGINE_LOGGER_ERROR("sp_open_camera failed!");
      return false;
    }
    FLOWENGINE_LOGGER_INFO("sp_open_camera is successed!");
    int yuv_size = FRAME_BUFFER_SIZE(mOptions.width, mOptions.height);
    yuv_data = reinterpret_cast<char *>(malloc(yuv_size * sizeof(char)));
    mStreaming.store(true);
    return true;
  };

  /**
   * Close the stream.
   * @see videoSource::Close()
   */
  virtual inline void Close() noexcept override {
    sp_vio_close(camera);
    free(yuv_data);
    mStreaming.store(false);
  }

  /**
   * Return the interface type
   */
  virtual inline size_t GetType() const noexcept override { return Type; }

  /**
   * Unique type identifier of camera class.
   */
  static const size_t Type = (1 << 0);

private:
  XCamera(videoOptions const &options) : videoSource(options) {}
  void *camera;
  char *yuv_data;

  virtual bool Init() override {
    camera = sp_init_vio_module();
    return true;
  }
};

} // namespace video
#endif