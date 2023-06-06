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
#include <cassert>
#include <opencv2/imgcodecs.hpp>

namespace module {
void FrameDifferenceModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} FrameDifferenceModule was done!", name);
      stopFlag.store(true);
      return;
    }
    auto frameBufMessage = ptr->pool->read(buf.key);
    auto frame =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

    // 只能有一个区域被监控
    assert(config->regions.size() == 1);
    auto &region = config->regions.at(0);
    int x = region[0].x;
    int y = region[0].y;
    int w = region[1].x - x;
    int h = region[1].y - y;
    cv::Rect rect{x, y, w, h};
    // 截取待计算区域
    cv::Mat croppedImage;
    infer::utils::cropImage(*frame, croppedImage, rect, buf.frameType);

    // 转成RGB类型去处理
    switch (buf.frameType) {
    case common::ColorType::NV12: {
      cv::cvtColor(croppedImage, croppedImage, cv::COLOR_YUV2RGB_NV12);
    }
    default: {
      break;
    }
    }
    std::vector<common::RetBox> bboxes;
    fd.update(croppedImage, bboxes); // 动态物检测
    if (bboxes.empty()) {
      return;
    }
    // 意味着检测到了动态目标，可以进行后续的逻辑（5进3或其他过滤方法）
    for (auto &bbox : bboxes) {
      // offset bbox
      bbox.second.at(0) += rect.x;
      bbox.second.at(1) += rect.y;
      bbox.second.at(2) += rect.x;
      bbox.second.at(3) += rect.y;
    }
    autoSend(buf);
  }
}
FlowEngineModuleRegister(FrameDifferenceModule, backend_ptr,
                         std::string const &, MessageType &, ModuleConfig &);
} // namespace module