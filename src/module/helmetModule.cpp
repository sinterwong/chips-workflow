/**
 * @file helmetModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-09-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "helmetModule.h"
#include <cstddef>
#include <cstdlib>
#include <experimental/filesystem>
#include <sys/stat.h>
#include <unistd.h>

namespace module {

/**
 * @brief
 *
 * @param message
 */
void HelmetModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} HelmetModule module was done!", name);
      stopFlag.store(true);
      return;
    }
  }
}

FlowEngineModuleRegister(HelmetModule, backend_ptr, std::string const &,
                         MessageType const &, LogicBase const &);
} // namespace module
