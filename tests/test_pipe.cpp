/**
 * @file detect_module.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-20
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <condition_variable>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "module.hpp"

#include "pipeline.hpp"

#include <gflags/gflags.h>
#include <unordered_map>
#include <vector>

DEFINE_string(config_path, "", "Specify config path.");
DEFINE_string(result_url, "", "Specify send result url.");
DEFINE_int32(num_workers, 5, "Specify number of thread pool .");

using namespace module;
// using common::WORKER_TYPES;

int main(int argc, char **argv) {
  FlowEngineLoggerInit(true, true, true, true);
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

#ifndef NX_SERIAL_NUMBER
  FLOWENGINE_LOGGER_ERROR("device error!");
  exit(-1);
#else
  std::ifstream sn_file("/proc/device-tree/serial-number");
  if (!sn_file.is_open()) {
    FLOWENGINE_LOGGER_ERROR("Failed to get device info!");
    return 0;
  }
  std::string tmp;
  std::string serial_number;
  while (getline(sn_file, tmp)) {
    serial_number += tmp;
  }
  sn_file.close();
  size_t n = serial_number.find_last_not_of(" \r\n\t");
  if (n != std::string::npos) {
    serial_number.erase(n + 1, serial_number.size() - n);
  }
  n = serial_number.find_first_not_of(" \r\n\t");
  if (n != std::string::npos) {
    serial_number.erase(0, n);
  }
  serial_number.erase(serial_number.end() - 1);  // 删除最后一位'^@'特殊符号

  if (serial_number != NX_SERIAL_NUMBER) {
    FLOWENGINE_LOGGER_ERROR("device error!");
    return -1;
  }
#endif

  std::shared_ptr<PipelineModule> pipeline(new PipelineModule(
      FLAGS_config_path, FLAGS_result_url, FLAGS_num_workers));
  pipeline->run();
  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}

/*
./test_pipe
--config_path=/home/wangxt/workspace/projects/flowengine/conf/app/config.json
--result_url=http://114.242.23.39:9400/v1/internal/receive_alarm
*/