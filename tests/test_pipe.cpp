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

  std::shared_ptr<PipelineModule> pipeline(new PipelineModule(FLAGS_config_path, FLAGS_result_url, FLAGS_num_workers));

  pipeline->go();

  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}

/*
./test_pipe --config_path=/home/wangxt/workspace/projects/flowengine/conf/app/config.json --result_url=http://114.242.23.39:9400/v1/internal/receive_alarm
*/