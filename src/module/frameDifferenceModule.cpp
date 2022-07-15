//
// Created by Wallel on 2022/3/1.
//

#include "frameDifferenceModule.h"
#include "backend.h"
#include "frame_difference.h"
#include "inference.h"
#include "logger/logger.hpp"
#include <cassert>
#include <opencv2/imgcodecs.hpp>

namespace module {
FrameDifferenceModule::FrameDifferenceModule(
    Backend *ptr, const std::string &initName, const std::string &initType,
    const std::vector<std::string> &recv, const std::vector<std::string> &send,
    const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool) {}

FrameDifferenceModule::~FrameDifferenceModule() {}

void FrameDifferenceModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      FLOWENGINE_LOGGER_INFO("FreameDifference module was done!");
      // std::cout << "FreameDifference module was done!" << std::endl;
      stopFlag.store(true);
    } else if (type == "FrameMessage") {
      auto frameBufMessage = backendPtr->pool.read(buf.key);
      auto frame =
          std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
      if (fd.statue()) {
        fd.init(frame);
      } else {
        fd.update(frame, buf.results.bboxes);
      }
      if (!buf.results.bboxes.empty()) {
        autoSend(buf);
      }
    }
  }
}
} // namespace module