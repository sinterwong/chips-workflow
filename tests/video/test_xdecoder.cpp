#include "logger/logger.hpp"
#include "gflags/gflags.h"
#include "videoDecode.hpp"

#include <cassert>
#include <cstddef>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace video;

DEFINE_string(uri, "", "Specify the url of video.");
DEFINE_int32(start, 0, "stream start index.");
DEFINE_int32(end, 9, "stream end index.");

void run_stream(std::string const &url, int idx) {

  // 视频流
  video::VideoDecode decoder{FLAGS_uri};

  decoder.init();
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");
  decoder.run();
  FLOWENGINE_LOGGER_INFO("Video manager is running!");

  std::string savePath = std::to_string(idx) + "_test_xdecoder.jpg";
  int count = 0;
  while (1) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    ++count;
    if (count % 20 != 0) {
      continue;
    }
    std::cout << count << ": " << decoder.getHeight() << ", " << decoder.getWidth()
              << std::endl;
    FLOWENGINE_LOGGER_INFO("id: {}, height: {}, width: {}, count: {}", idx,
                           decoder.getHeight(), decoder.getWidth(), count);
    auto nv12_image = decoder.getcvImage();
    if (!nv12_image->empty()) {
      std::cout << "saving the image" << std::endl;
      cv::imwrite(savePath, *nv12_image);
    }
  }
}

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  std::vector<std::string> streams{
      "rtsp://admin:zkfd123.com@114.242.23.39:9303/Streaming/Channels/101",
      "rtsp://admin:zkfd123.com@114.242.23.39:9304/Streaming/Channels/101",
      "rtsp://114.242.23.39:6001/MainStream",
      "rtsp://114.242.23.39:6002/MainStream",
      "rtsp://114.242.23.39:6004/MainStream",
      "rtsp://114.242.23.39:6006/MainStream",
      "rtsp://114.242.23.39:6008/MainStream",
      "rtsp://114.242.23.39:6554/live/test0",
      "rtsp://114.242.23.39:6554/live/test1",
      "rtsp://114.242.23.39:6554/live/test2",
  };

  assert(FLAGS_end <= int(streams.size()));

  std::vector<std::thread> worker_threads;

  for (int i = FLAGS_start; i < FLAGS_end; ++i) {
    worker_threads.emplace_back(std::thread(run_stream, streams.at(i), i));
  }

  for (auto &worker_thread : worker_threads) {
    worker_thread.join();
  }

  return 0;
}