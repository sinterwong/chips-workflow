/**
 * @file test_decoder_pool.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * 解码池可行性实验，本例的目标是使用一个解码器同时解码两路视频流同尺寸和编码类型的视频
 * @version 0.1
 * @date 2023-08-21
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "ffstream.hpp"
#include "logger/logger.hpp"
#include <exception>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sp_codec.h>

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

using video::utils::FFStream;

bool set_and_get_data(std::shared_ptr<FFStream> stream, void *decoder,
                      char *yuv_data, int channel, bool first = false) {
  void *raw_data; 
  // 只取I帧的情况下可以做到一路解码处理多路视频流
  int bufSize = stream->getRawFrame(&raw_data, false, true);
  if (bufSize < 0)
    std::terminate();
  if (bufSize == 0) {
    // 不是I帧
    return false;
  }
  // 送入解码器数据
  int ret = sp_decoder_set_image(decoder, reinterpret_cast<char *>(raw_data),
                                 channel, bufSize, 0);
  if (ret != 0) {
    FLOWENGINE_LOGGER_WARN("sp_decoder_set_image is failed: {}", ret);
    return false;
  }

  if (first) {
    return true;
  }

  ret = sp_decoder_get_image(decoder, yuv_data);
  if (ret != 0) {
    FLOWENGINE_LOGGER_WARN("sp_decoder_get_image get next frame is failed!");
    return false;
  }

  return true;
}

int main(int argc, char **argv) {
  std::string url1 =
      "rtsp://admin:ghrq123456@114.242.23.39:6104/Streaming/Channels/101";
  std::string url2 =
      "rtsp://admin:ghrq123456@114.242.23.39:6104/Streaming/Channels/201";

  // 解码器handle
  void *decoder;
  // 初始化解码器
  decoder = sp_init_decoder_module();

  // 设置两个视频流
  auto stream1 = std::make_shared<FFStream>(url1);
  auto stream2 = std::make_shared<FFStream>(url2);

  if (!stream1->openStream()) {
    FLOWENGINE_LOGGER_ERROR("can't open the stream {}!", std::string(url1));
    return -1;
  }
  if (!stream2->openStream()) {
    FLOWENGINE_LOGGER_ERROR("can't open the stream {}!", std::string(url2));
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("opening the streams has successed!");

  // 启动解码器
  int ret = sp_start_decode(decoder, "", 0, SP_ENCODER_H265,
                            stream1->getWidth(), stream1->getHeight());
  if (ret != 0) {
    FLOWENGINE_LOGGER_ERROR("opening sp_open_decoder has failed {}!", ret);
    return -1;
  }
  FLOWENGINE_LOGGER_INFO("sp_open_decoder has successed!");

  // 解码后数据
  size_t yuv_size = stream1->getWidth() * stream1->getHeight() * 3 / 2;
  char *yuv_data1;
  yuv_data1 = new char[yuv_size];
  char *yuv_data2;
  yuv_data2 = new char[yuv_size];

  // // 扔掉stream2中的第一帧
  // void *temp_d1;
  // int firstPacket1 = stream2->getRawFrame(&temp_d1);
  // FLOWENGINE_LOGGER_INFO("Stream2 size of the first packet is {}",
  //                        firstPacket1);

  // // 配置第一帧为stream1的头
  // auto ret1 = set_and_get_data(stream1, decoder, yuv_data1, 0, true);
  // if (ret1) {
  //   FLOWENGINE_LOGGER_INFO("Setting first packet has successed!");
  // }

  int count = 0;
  while (count++ < 100) {
    auto ret1 = set_and_get_data(stream1, decoder, yuv_data1, 0);
    if (ret1) {
      // 解码成功，打印图片
      auto image1_nv12 = cv::Mat(stream1->getHeight() * 3 / 2,
                                 stream1->getWidth(), CV_8UC1, yuv_data1);
      cv::Mat image1_bgr;
      cv::cvtColor(image1_nv12, image1_bgr, cv::COLOR_YUV2BGR_NV12);
      cv::imwrite("test_decoder_pool_image1_bgr.jpg", image1_bgr);
    }
    auto ret2 = set_and_get_data(stream2, decoder, yuv_data2, 0);
    if (ret2) {
      // 解码成功，打印图片
      auto image2_nv12 = cv::Mat(stream1->getHeight() * 3 / 2,
                                 stream1->getWidth(), CV_8UC1, yuv_data2);
      cv::Mat image2_bgr;
      cv::cvtColor(image2_nv12, image2_bgr, cv::COLOR_YUV2BGR_NV12);
      cv::imwrite("test_decoder_pool_image2_bgr.jpg", image2_bgr);
    }
  }

  sp_release_decoder_module(decoder);
  return 0;
}