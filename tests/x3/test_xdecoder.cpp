#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include "x3/ffstream.hpp"
#include <chrono>
#include <iostream>
#include <sp_codec.h>
#include <string>
#include <utility>
using namespace std::chrono_literals;
using namespace module::utils;

int main(int argc, char **argv) {
  std::string uri =
      "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101";
  std::unique_ptr<FFStream> stream = std::make_unique<FFStream>(uri);
  if (!stream->openStream()) {
    FLOWENGINE_LOGGER_ERROR("can't open the stream {}!", std::string(uri));
    return -1;
  }
  void *decoder = sp_init_decoder_module();
  void *decoder2 = sp_init_decoder_module();
  if (!decoder2) {
    return -1; 
  }
  int ret = sp_start_decode(decoder, "", 0, SP_ENCODER_H264, 1920, 1080);
  if (ret != 0) {
    return -1;
  }
  // void *raw_data;
  // char *yuv_data = reinterpret_cast<char *>(
  //     malloc(stream->getHeight() * 3 / 2 * stream->getWidth() * sizeof(char)));
  // while (stream->isRunning()) {
  //   int bufSize;
  //   bufSize = stream->getRawFrame(&raw_data);
  //   if (bufSize < 0) {
  //     bufSize = 0;
  //   }
  //   int ret = sp_decoder_set_image(decoder, reinterpret_cast<char *>(&raw_data),
  //                                  0, bufSize, 0);
  //   if (ret != 0) {
  //     FLOWENGINE_LOGGER_WARN("sp_decoder_set_image is failed: {}", ret);
  //     std::this_thread::sleep_for(2s);
  //   }
  //   ret = sp_decoder_get_image(decoder, yuv_data);
  //   if (ret != 0) {
  //     FLOWENGINE_LOGGER_WARN("sp_decoder_get_image get next frame is failed!");
  //     std::this_thread::sleep_for(100ms);
  //   } else {
  //     cv::Mat nv12_image = cv::Mat(stream->getHeight() * 3 / 2,
  //                                  stream->getWidth(), CV_8UC1, yuv_data);
  //     cv::imwrite("test_xdecoder.jpg", nv12_image);
  //   }
  // }
  std::cout << "successfully" << std::endl;
  return 0;
}