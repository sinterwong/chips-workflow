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
#include "video_utils.hpp"
#include <algorithm>
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
    // 读取图片
    frame_ptr frameBuf = ptr->pools->read(buf.steramName, buf.key);
    if (!frameBuf) {
      FLOWENGINE_LOGGER_WARN("{} FrameDifferenceModule read frame is failed!",
                             name);
      return;
    }
    auto frame = std::any_cast<std::shared_ptr<cv::Mat>>(frameBuf->read("Mat"));

    // 只能有一个区域被监控
    assert(config->regions.size() <= 1);
    cv::Mat croppedImage;
    cv::Rect rect;
    if (config->regions.size() == 1) {
      auto &points = config->regions.at(0);
      int x = std::min(points[0].x, points[1].x);
      int y = std::min(points[0].y, points[1].y);
      int w = std::max(points[0].x, points[1].x) - x;
      int h = std::max(points[0].y, points[1].y) - y;
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
      bbox.x += rect.x;
      bbox.y += rect.y;
      // bbox.second.at(2) += rect.x;
      // bbox.second.at(3) += rect.y;
    }
    FLOWENGINE_LOGGER_DEBUG("{} FrameDifferenceModule detect {} objects", name,
                            bboxes.size());
    // 生成报警信息
    alarmUtils.generateAlarmInfo(name, buf.alarmInfo, "存在报警行为",
                                 config.get());
    // 生成报警图片
    alarmUtils.saveAlarmImage(
        buf.alarmInfo.alarmFile + "/" + buf.alarmInfo.alarmId + ".jpg", *frame,
        buf.frameType, static_cast<DRAW_TYPE>(config->drawType));
    autoSend(buf);
    // 录制报警视频
    if (config->videoDuration > 0) {
      bool ret = video::utils::videoRecordWithFFmpeg(
          buf.alarmInfo.cameraIp,
          buf.alarmInfo.alarmFile + "/" + buf.alarmInfo.alarmId + ".mp4",
          config->videoDuration);
      if (!ret) {
        FLOWENGINE_LOGGER_ERROR("{} video record is failed.", name);
      }
    }

    break;
  }
}
FlowEngineModuleRegister(FrameDifferenceModule, backend_ptr,
                         std::string const &, MessageType &, ModuleConfig &);
} // namespace module