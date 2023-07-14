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
#include "logger/logger.hpp"
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

    if (alarmUtils.isRecording()) { // 正在录像
      alarmUtils.recordVideo(*frame);
      break;
    }

    // 只能有一个区域被监控
    assert(config->regions.size() <= 1);
    cv::Mat croppedImage;
    cv::Rect rect;
    if (config->regions.size() == 1) {
      auto &region = config->regions.at(0);
      int x = region[0].x;
      int y = region[0].y;
      int w = region[1].x - x;
      int h = region[1].y - y;
      rect = {x, y, w, h};
      // 截取待计算区域
      infer::utils::cropImage(*frame, croppedImage, rect, buf.frameType);
    } else {
      croppedImage = frame->clone();
      rect = {0, 0, croppedImage.cols, croppedImage.rows};
    }

    // 转成RGB类型去处理
    switch (buf.frameType) {
    case common::ColorType::NV12: {
      cv::cvtColor(croppedImage, croppedImage, cv::COLOR_YUV2RGB_NV12);
    }
    default: {
      break;
    }
    }
    // 动态物检测  TODO 针对某些任务可以通过resize来提高效率（如摄像头偏移检测）
    std::vector<common::RetBox> bboxes;
    fd.update(croppedImage, bboxes, config->threshold);
    if (bboxes.empty()) {
      break;
    }
    // 意味着检测到了动态目标，可以进行后续的逻辑（5进3或其他过滤方法，当前只是用了阈值）
    for (auto &bbox : bboxes) {
      // offset bbox
      bbox.second.at(0) += rect.x;
      bbox.second.at(1) += rect.y;
      bbox.second.at(2) += rect.x;
      bbox.second.at(3) += rect.y;
    }
    FLOWENGINE_LOGGER_DEBUG("{} FrameDifferenceModule detect {} objects", name,
                            bboxes.size());
    // 生成报警信息
    alarmUtils.generateAlarmInfo(name, buf.alarmInfo, "存在报警行为",
                                 config.get());
    // 生成报警图片
    alarmUtils.saveAlarmImage(buf.alarmInfo.alarmFile + "/" +
                                  buf.alarmInfo.alarmId + ".jpg",
                              *frame, buf.frameType, config->isDraw);
    // 初始化报警视频
    if (config->videoDuration > 0) {
      alarmUtils.initRecorder(
          buf.alarmInfo.alarmFile + "/" + buf.alarmInfo.alarmId + ".mp4",
          buf.alarmInfo.width, buf.alarmInfo.height, 25, config->videoDuration);
    }

    autoSend(buf);
    break;
  }
  if (!alarmUtils.isRecording()) {
    std::this_thread::sleep_for(std::chrono::microseconds{config->interval});
  }
}
FlowEngineModuleRegister(FrameDifferenceModule, backend_ptr,
                         std::string const &, MessageType &, ModuleConfig &);
} // namespace module