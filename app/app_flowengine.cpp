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

#include <fstream>
#include <iostream>
#include <memory>

#include <gflags/gflags.h>
#include <vector>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include "module/pipeline.hpp"

#include "license/licenseVerifier.hpp"

DEFINE_string(config_path, "", "Specify config path.");
DEFINE_int32(num_workers, 16, "Specify number of thread pool .");
DEFINE_string(flowengine_conf, "flowengine_conf",
              "Specify config file of flowengine .");

using namespace module;
// using common::WORKER_TYPES;

const auto initLogger = []() -> decltype(auto) {
  FlowEngineLoggerInit(true, true, true, true);
  FlowEngineLoggerSetLevel(2);
  return true;
}();

int main(int argc, char **argv) {
  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  /*
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
  */
#ifndef _DEBUG
  srand(time(0));
  // Read Conf File
  // Read Conf File
  std::ifstream in(FLAGS_flowengine_conf, std::ios::binary);
  if (!in) {
    in.close();
    std::cout << "Read Conf file failed." << std::endl;
    return -2;
  }

  int addr_num;
  std::string trialFile;
  std::string dstIDFile; // Device Tree Serial Num

  in.read((char *)&addr_num, sizeof(int));
  if (addr_num != 2) {
    in.close();
    std::cout << "Conf file error." << std::endl;
    return -2;
  }

  // Begin reading address
  fe_license::readConfOnce(in, trialFile);
  fe_license::readConfOnce(in, dstIDFile);
  in.close();

  int ret = fe_license::checkLicense("licenseA", "licenseB", "licenseC",
                                     dstIDFile.c_str());
  if (ret != 0) {
    // Check Trial
    int trailRet = fe_license::checkTrial(trialFile.c_str());
    std::cout << trailRet << std::endl;
    if (fe_license::checkTrial(trialFile.c_str())) {
      FLOWENGINE_LOGGER_CRITICAL("Your license has expired!");

      return -6;
    }
    return ret;
  }
#endif
  std::shared_ptr<PipelineModule> pipeline(
      std::make_shared<PipelineModule>(FLAGS_config_path, FLAGS_num_workers));
  pipeline->run();
  gflags::ShutDownCommandLineFlags();
  FlowEngineLoggerDrop();
  return 0;
}

/*
 ./app_flowengine --config_path=/public/agent/conf/agent.json --num_workers=24
*/