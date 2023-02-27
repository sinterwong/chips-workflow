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

CallingModule::CallingModule(backend_ptr ptr, std::string const &name,
                             std::string const &type, LogicConfig const &config)
    : LogicModule(ptr, name, type, config) {}

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

FlowEngineModuleRegister(CallingModule, backend_ptr, std::string const &,
                         std::string const &, LogicConfig const &);
} // namespace module
