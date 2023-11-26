/**
 * @file test_bandwidth_for_rtsp.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-08-25
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <mutex>

#include "ffstream.hpp"
#include "logger/logger.hpp"

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  return true;
}();

DEFINE_string(file_path, "", "Specify the file of url path.");

std::atomic<int> error_count(0);

bool rtsp_read_thread(std::string &url) {
  video::utils::FFStream stream(url);
  if (!stream.openStream()) {
    error_count++;
    return false;
  }

  void *data;
  // 每个线程读取100次
  while (stream.isRunning()) {
    int size = stream.getRawFrame(&data);
    if (size == -1) {
      error_count++;
      return false;
    }
  }
  return true;
}

std::vector<std::string> streams;                 // 所有的rtsp链接
std::unordered_map<std::string, int> usage_count; // 记录每个RTSP链接的使用次数
std::mutex mtx; // 用于确保线程安全的修改usage_count

std::string get_next_stream() {
  std::lock_guard<std::mutex> lock(mtx);
  for (const auto &url : streams) {
    if (usage_count[url] < 5) {
      usage_count[url]++;
      return url;
    }
  }
  return ""; // 如果所有链接都已经使用超过3次，返回一个空字符串
}

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

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

  for (const auto &url : streams) {
    usage_count[url] = 0; // 初始化每个链接的使用次数为0
  }

  if (streams.empty()) {
    FLOWENGINE_LOGGER_DEBUG("streams is Empty!");
    return 0;
  }

  int thread_count = 0;
  std::vector<std::thread> threads;
  while (true) {
    error_count = 0;
    std::string url = get_next_stream();
    if (url.empty()) {
      FLOWENGINE_LOGGER_DEBUG("streams had exhausted!");
      break; // 所有的链接使用都已经超过3次
    }
    threads.emplace_back(std::thread(rtsp_read_thread, std::ref(url)));
    thread_count++;

    // 让新的线程运行一段时间，以确保它有机会启动并读取流
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 如果有超过10%的线程报错，我们可以认为带宽已经撑不住
    if ((static_cast<float>(error_count) / thread_count) > 0.1) {
      FLOWENGINE_LOGGER_CRITICAL(
          "Network bandwidth is likely maxed out at {} threads", thread_count);
      break;
    }
  }
  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  gflags::ShutDownCommandLineFlags();
  return 0;
}