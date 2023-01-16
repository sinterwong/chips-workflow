/**
 * @file videoManager.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-12-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "videoManager.hpp"
#include "logger/logger.hpp"
#include "videoDecoder.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace module::utils {

bool VideoManager::init() {
  if (!reader->openStream()) {
    FLOWENGINE_LOGGER_ERROR("can't open the stream!");
    return false;
  }
  FLOWENGINE_LOGGER_INFO("{} is opened!", uri);
  FLOWENGINE_LOGGER_INFO("The video size is {}x{}, and codec is {}!",
                         reader->getHeight(), reader->getWidth(),
                         reader->getCodecType());

  vp_param.mmz_size = reader->getHeight() * reader->getWidth();

  memset(&stFrameInfo, 0, sizeof(VIDEO_FRAME_S));

  // 创建内存buff
  if (!x3_vp_alloc()) {
    FLOWENGINE_LOGGER_ERROR("x3_vp_alloc is failed");
    return false;
  }
  FLOWENGINE_LOGGER_INFO("x3_vp_alloc is successed!");

  if (!decoder->init(reader->getCodecType(), reader->getWidth(),
                     reader->getHeight())) {
    FLOWENGINE_LOGGER_ERROR("decoder init is failed!");
    return false;
  }
  FLOWENGINE_LOGGER_INFO("decoder initialization is successed!");
  return true;
}

void VideoManager::streamSend() {
  // {
  //   std::lock_guard<std::mutex> lk(m);
  //   // video decoder init
  //   if (!decoder->init(reader->getCodecType(), reader->getWidth(),
  //                      reader->getHeight())) {
  //     FLOWENGINE_LOGGER_ERROR("decoder init is failed!");
  //   }
  //   FLOWENGINE_LOGGER_INFO("decoder initialization is successed!");
  // }
  // is_start.notify_all();
  while (true) {
    if (reader->isRunning() && decoder->isRunning()) {
      mmzIndex = reader->getParam().count % vp_param.mmz_cnt;
      int bufSize = reader->getRawFrame(
          reinterpret_cast<void *>(vp_param.mmz_vaddr[mmzIndex]));
      if (bufSize < 0) {
        bufSize = 0;
      }
      // std::cout << reader->getParam() << std::endl;
      if (!decoder->sendStream(mmzIndex, reader->getParam().count,
                               vp_param.mmz_paddr[mmzIndex],
                               &vp_param.mmz_vaddr[mmzIndex], bufSize)) {
        // break;
        std::this_thread::sleep_for(2s);
      }
    } else {
      std::this_thread::sleep_for(2s);
    }
  }
  FLOWENGINE_LOGGER_WARN("reader status is {}, and decoder status is {}",
                         reader->isRunning(), decoder->isRunning());
  FLOWENGINE_LOGGER_WARN("VideoManager: streamSend is over!");
}

void VideoManager::streamGet() {
  // std::unique_lock<std::mutex> lk(m);
  // is_start.wait(lk, [this] { return !decoder->isRunning(); });
  while (true) {
    if (decoder->isRunning()) {
      std::lock_guard<std::mutex> lk(m);
      decoder->getFrame(stFrameInfo);
    } else {
      std::this_thread::sleep_for(2s);
    }
  }
}

bool VideoManager::run() {
  // recv = std::make_unique<joining_thread>(&VideoManager::streamGet, this);
  recv = std::make_unique<std::thread>(&VideoManager::streamGet, this);
  std::this_thread::sleep_for(2ms);
  send = std::make_unique<std::thread>(&VideoManager::streamSend, this);
  return true;
}

cv::Mat VideoManager::getcvImage() {
  std::lock_guard lk(m);
  // 需要一个能判定stFrameInfo是否有值的条件
  // std::cout << stFrameInfo.stVFrame.size << std::endl;
  if (stFrameInfo.stVFrame.size < 1) {
    return cv::Mat();
  }
  int height = static_cast<int>(stFrameInfo.stVFrame.height);
  int width = static_cast<int>(stFrameInfo.stVFrame.width);

  return cv::Mat(height * 3 / 2, width, CV_8UC1,
                 stFrameInfo.stVFrame.vir_ptr[0])
      .clone();
}

} // namespace module::utils