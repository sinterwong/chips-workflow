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
#include "frame_difference.h"
#include <opencv2/imgcodecs.hpp>

namespace module {
FrameDifferenceModule::FrameDifferenceModule(backend_ptr ptr,
                                             std::string const &name,
                                             MessageType const &type)
    : Module(ptr, name, type), fd(name) {}

FrameDifferenceModule::~FrameDifferenceModule() {}

void FrameDifferenceModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("FreameDifference module was done!");
      stopFlag.store(true);
      return;
    }
    auto frameBufMessage = ptr->pool->read(buf.key);
    auto frame =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
    std::vector<common::RetBox> bboxes;
    if (fd.statue()) {
      fd.init(frame);
    } else {
      fd.update(frame, bboxes);
    }

    if (bboxes.empty()) {
      return;
    }
    // 意味着检测到了动态目标，可以进行后续的逻辑
    // autoSend(buf);
  }
}
FlowEngineModuleRegister(FrameDifferenceModule, backend_ptr,
                         std::string const &, MessageType &);
} // namespace module