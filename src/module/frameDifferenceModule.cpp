/**
 * @file frameDifferenceModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-15
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "frameDifferenceModule.h"
#include "backend.h"
#include "frame_difference.h"
#include "inference.h"
#include "logger/logger.hpp"
#include "module.hpp"
#include <cassert>
#include <opencv2/imgcodecs.hpp>

namespace module {
FrameDifferenceModule::FrameDifferenceModule(backend_ptr ptr,
                                             std::string const &name,
                                             std::string const &type)
    : Module(ptr, name, type), fd(name) {}

FrameDifferenceModule::~FrameDifferenceModule() {}

void FrameDifferenceModule::forward(std::vector<forwardMessage> &message) {
  // for (auto &[send, type, buf] : message) {
  //   if (type == "ControlMessage") {
  //     // FLOWENGINE_LOGGER_INFO("FreameDifference module was done!");
  //     std::cout << "FreameDifference module was done!" << std::endl;
  //     stopFlag.store(true);
  //     return;
  //   } else if (type == "stream") {
  //     auto frameBufMessage = ptr->pool->read(buf.key);
  //     auto frame =
  //         std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
  //     if (fd.statue()) {
  //       fd.init(frame);
  //     } else {
  //       fd.update(frame, buf.algorithmResult.bboxes);
  //     }
  //     if (!buf.algorithmResult.bboxes.empty()) {
  //       autoSend(buf);
  //     }
  // }
  // }
}
FlowEngineModuleRegister(FrameDifferenceModule, backend_ptr,
                         std::string const &, std::string const &);
} // namespace module