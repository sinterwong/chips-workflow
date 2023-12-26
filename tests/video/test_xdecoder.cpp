#include "gflags/gflags.h"
#include "logger/logger.hpp"
#include "videoDecode.hpp"

#include <cassert>
#include <cstddef>
#include <fstream>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using namespace video;

DEFINE_string(file_path, "", "Specify the file of url path.");
DEFINE_int32(start, 0, "stream start index.");
DEFINE_int32(end, -1, "stream end index.");
DEFINE_int32(count, 1000, "run times.");

const bool initLogger = []() {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

void run_stream(std::string const &url, int idx) {

  // 视频流
  video::VideoDecode decoder;

  if (!decoder.init()) {
    FLOWENGINE_LOGGER_INFO("{} has initialized failed!", idx);
  }
  FLOWENGINE_LOGGER_INFO("{} has initialized!", idx);
  if (!decoder.start(url)) {
    FLOWENGINE_LOGGER_ERROR("{} has run failed!", idx);
    return;
  }
  FLOWENGINE_LOGGER_INFO("{} is running!", idx);

  std::string savePath = std::to_string(idx) + "_test_xdecoder.jpg";
  int count = 0;
  while (count < FLAGS_count) {
    if (count % 20 == 0) {
      std::cout << count << ": " << decoder.getHeight() << ", "
                << decoder.getWidth() << std::endl;
      FLOWENGINE_LOGGER_INFO("id: {}, height: {}, width: {}, count: {}", idx,
                             decoder.getHeight(), decoder.getWidth(), count);
    }
    auto nv12_image = decoder.getcvImage();
    if (nv12_image && !nv12_image->empty()) {
      cv::imwrite(savePath, *nv12_image);
    } else {
      FLOWENGINE_LOGGER_ERROR("{} get image was failed!", idx);
    }
    ++count;
  }
}

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> streams;
  std::ifstream url_file(FLAGS_file_path);
  if (!url_file.is_open()) {
    FLOWENGINE_LOGGER_ERROR("Failed to open file {}", FLAGS_file_path);
    return -1;
  }
  // 逐行读取数据
  std::string line;
  while (getline(url_file, line)) {
    line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
    if (!line.empty()) {
      streams.push_back(line);
    }
  }
  url_file.close();

  int start = 0, end;
  if (FLAGS_start > 0) {
    start = FLAGS_start;
  }
  if (FLAGS_end < 0) {
    end = streams.size();
  } else {
    end = FLAGS_end;
  }
  std::vector<std::thread> worker_threads;

  for (int i = start; i < end; ++i) {
    worker_threads.emplace_back(std::thread(run_stream, streams.at(i), i));
  }

  for (auto &worker_thread : worker_threads) {
    worker_thread.join();
  }

  return 0;
}