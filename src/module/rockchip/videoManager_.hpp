/**
 * @file videoManager.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-12-02
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __VIDEO_MANAGER_FOR_X3_OLD_H_
#define __VIDEO_MANAGER_FOR_X3_OLD_H_
#include "common/common.hpp"
#include "ffmstream.hpp"
#include "hb_vp_api.h"
#include "joining_thread.h"
#include "logger/logger.hpp"
#include "videoDecoder.hpp"
#include <condition_variable>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <thread>

namespace module::utils {

class VideoManager : private common::NonCopyable {
private:
  struct vp_param_t {
    uint64_t mmz_paddr[3];
    char *mmz_vaddr[3];
    int mmz_cnt = 3;
    int mmz_size;
  };

  inline bool x3_vp_alloc() {
    int i = 0, ret = 0;
    for (i = 0; i < vp_param.mmz_cnt; i++) {
      ret = HB_SYS_Alloc(&vp_param.mmz_paddr[i],
                         (void **)&vp_param.mmz_vaddr[i], vp_param.mmz_size);
      if (!ret) {
        FLOWENGINE_LOGGER_INFO(
            "mmzAlloc paddr = 0x{}, vaddr = 0x{}, {}/{} , {}",
            vp_param.mmz_paddr[i], vp_param.mmz_vaddr[i], i, vp_param.mmz_cnt,
            vp_param.mmz_size);
      } else {
        FLOWENGINE_LOGGER_ERROR("hb_vp_alloc failed, ret: {}", ret);
        return false;
      }
    }
    return true;
  }

  inline bool x3_vp_free() {
    int i = 0, ret = 0;
    for (i = 0; i < vp_param.mmz_cnt; i++) {
      ret = HB_SYS_Free(vp_param.mmz_paddr[i], vp_param.mmz_vaddr[i]);
      if (ret == 0) {
        FLOWENGINE_LOGGER_INFO("mmzFree paddr = 0x{}, vaddr = 0x{} i = {}",
                               vp_param.mmz_paddr[i], vp_param.mmz_vaddr[i], i);
      }
    }
    return true;
  }

  inline bool memory_alloc() {
    // TODO 尝试内部模式，分配普通内存即可
    return true;
  }

  inline bool memory_dealloc() {
    // TODO 尝试内部模式，分配普通内存即可
    return true;
  }
  
  // stream manager
  std::unique_ptr<FmpStream> reader;
  std::unique_ptr<VideoDecoder> decoder;
  std::string uri; // 流地址
  int videoId;     // 编码通道
  int mmzIndex;    // 循环索引

  // 获取解码之后的数据
  VIDEO_FRAME_S stFrameInfo;

  // vp_param_t vp_param;
  vp_param_t vp_param;
  std::unique_ptr<std::thread> send; // 生产者
  std::unique_ptr<std::thread> recv; // 消费者
  std::mutex m;
  // std::condition_variable is_start;
  // std::shared_ptr<cv::Mat> sharedImage;

  void streamSend();

  void streamGet();

public:
  bool init();

  void run();

  inline bool isRunning() {
    return reader->isRunning() && decoder->isRunning();
  }

  inline int getHeight() { return reader->getHeight(); }

  inline int getWidth() { return reader->getWidth(); }

  inline int getRate() { return reader->getRate(); }

  inline void join() noexcept {
    recv->join();
    send->join();
  }

  cv::Mat getcvImage();

  inline common::ColorType getType() const noexcept {
    return common::ColorType::NV12;
  }

  explicit VideoManager(std::string const &uri_, int idx_) noexcept
      : uri(uri_), videoId(idx_) {
    reader = std::make_unique<FmpStream>(uri_);
    decoder = std::make_unique<VideoDecoder>(idx_, vp_param.mmz_cnt);
  }

  ~VideoManager() noexcept {
    join();
    x3_vp_free();
  }
};
} // namespace module::utils

#endif