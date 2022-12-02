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

namespace module::utils {

bool VideoManager::init() {
  if (!reader->openStream()) {
    FLOWENGINE_LOGGER_ERROR("can't open the stream!");
    return false;
  }
  FLOWENGINE_LOGGER_ERROR("{} is opened!", uri);

  vp_param.mmz_size = reader->getHeight() * reader->getWidth();

  memset(&stFrameInfo, 0, sizeof(VIDEO_FRAME_S));

  // 创建内存buff
  if (!x3_vp_alloc()) {
    FLOWENGINE_LOGGER_ERROR("vdec_ChnAttr_init failed, {}");
    return false;
  }

  // video decoder init
  if (!decoder->init(reader->getCodecType(), reader->getWidth(),
                     reader->getHeight())) {
    FLOWENGINE_LOGGER_ERROR("decoder init is failed!");
  }

  return true;
}

void VideoManager::streamSend() {
  while (reader->isRunning() && decoder->isRunning()) {
    mmzIndex = reader->getParam().count % vp_param.mmz_cnt;
    int bufSize = reader->getRawFrame(
        reinterpret_cast<void *>(vp_param.mmz_vaddr[mmzIndex]));
    if (bufSize > 0) {
      bufSize = 0;
    }
    if (!decoder->sendStream(mmzIndex, reader->getParam().count,
                             vp_param.mmz_paddr[mmzIndex],
                             vp_param.mmz_vaddr[mmzIndex], bufSize)) {
      break;
    }
  }
}

void VideoManager::streamGet() {
  while (decoder->isRunning()) {
    std::lock_guard<std::mutex> lk(m);
    decoder->getFrame(stFrameInfo);
  }
}

void VideoManager::run() {
  send = std::make_unique<joining_thread>(&VideoManager::streamSend, this);
  recv = std::make_unique<joining_thread>(&VideoManager::streamGet, this);
}

std::shared_ptr<cv::Mat> VideoManager::getcvImage() {
  std::lock_guard lk(m);
  if (!stFrameInfo.stVFrame
           .vir_ptr[0]) { // 需要一个能判定stFrameInfo是否有值的条件
    return std::shared_ptr<cv::Mat>();
  }
  int height = static_cast<int>(stFrameInfo.stVFrame.height);
  int width = static_cast<int>(stFrameInfo.stVFrame.width);
  sharedImage = std::make_shared<cv::Mat>(
      cv::Mat(height * 3 / 2, width, CV_8UC1, stFrameInfo.stVFrame.vir_ptr[0])
          .clone());
  return sharedImage;
}

} // namespace module::utils