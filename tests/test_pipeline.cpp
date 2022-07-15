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
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "module.hpp"

#include "controlModule.hpp"

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

  std::shared_ptr<ControlModule> control(new ControlModule(FLAGS_config_path, FLAGS_result_url, FLAGS_num_workers));
  if (!control->initialize()) {
    return -1;
  }

  control->go();

  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}

/*
./test_pipeline --config_path /home/wangxt/workspace/projects/flowengine/conf/app/114_242_23_39_9303.json --result_url http://114.242.23.39:9400/v1/internal/receive_alarm
*/