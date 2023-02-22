/**
 * @file CallingModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-07-31
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "callingModule.h"
#include "logger/logger.hpp"
#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

namespace module {

CallingModule::CallingModule(Backend *ptr, const std::string &initName,
                             const std::string &initType,
                             const common::LogicConfig &logicConfig)
    : LogicModule(ptr, initName, initType, logicConfig) {}

/**
 * @brief
 * 1. recv 类型：stream, algorithm
 * 2. send 类型：algorithm, output
 *
 * @param message
 */
void CallingModule::forward(std::vector<forwardMessage> &message) {
  if (recvModule.empty()) {
    return;
  }
}

FlowEngineModuleRegister(CallingModule, Backend *, std::string const &,
                         std::string const &, common::LogicConfig const &);
} // namespace module
