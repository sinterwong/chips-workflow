/**
 * @file objectNumber.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-04-18
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "objectNumberModule.h"
#include "logger/logger.hpp"
#include "video_utils.hpp"

#include <cassert>

namespace module {

/**
 * @brief
 *
 * @param message
 */
void ObjectNumberModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} ObjectNumberModule was done!", name);
      stopFlag.store(true);
      return;
    }

    // 读取图片
    frame_ptr frameBuf = ptr->pools->read(buf.steramName, buf.key);
    if (!frameBuf) {
      FLOWENGINE_LOGGER_WARN("{} ObjectNumberModule read frame is failed!",
                             name);
      return;
    }
    auto image = std::any_cast<std::shared_ptr<cv::Mat>>(frameBuf->read("Mat"));

    // 初始待计算区域，每次算法结果出来之后需要更新regions
    std::vector<common::RetBox> regions;
    for (auto const &area : config->regions) {
      regions.emplace_back(RetBox(name, area));
    }
    if (regions.empty()) {
      // 前端没有画框
      regions.emplace_back(common::RetBox{name});
    }

    // 根据提供的配置执行算法，
    auto &detNet = config->algoPipelines.at(0);
    auto &attentions = detNet.second.attentions; // 检测关注的类别

    int objectNumber = 0;
    std::vector<RetBox> rbboxes; // 报警框

    for (auto &region : regions) {

      InferParams detParams{name,
                            buf.frameType,
                            detNet.second.cropScaling,
                            region,
                            {image->cols, image->rows, image->channels()}};

      InferResult detRet;
      if (!ptr->algo->infer(detNet.first, image->data, detParams, detRet)) {
        return;
      };
      // 获取检测结果信息
      auto bboxes = std::get_if<common::BBoxes>(&detRet.aRet);
      if (!bboxes) {
        FLOWENGINE_LOGGER_ERROR("ObjectNumberModule: Wrong algorithm type!");
        return;
      }
      for (auto &bbox : *bboxes) {
        // 需要过滤掉不关注的类别
        if (!attentions.empty()) {
          auto iter = std::find(attentions.begin(), attentions.end(),
                                static_cast<int>(bbox.class_id));
          if (iter == attentions.end()) { // 该类别没有出现在关注类别中
            continue;
          }
          // // 阈值太低的过滤
          // if (bbox.det_confidence < 0.6) {
          //   continue;
          // }
        }
        common::RetBox b = {name,
                            static_cast<int>(bbox.bbox[0] + region.x),
                            static_cast<int>(bbox.bbox[1] + region.y),
                            static_cast<int>(bbox.bbox[2] - bbox.bbox[0]),
                            static_cast<int>(bbox.bbox[3] - bbox.bbox[1]),
                            bbox.det_confidence,
                            static_cast<int>(bbox.class_id)};
        rbboxes.emplace_back(std::move(b));
        objectNumber++;
      }
    }

    // 每到一定的数量就会触发报警
    if (objectNumber >= config->amount) {
      FLOWENGINE_LOGGER_DEBUG("object number: {}", objectNumber);

      // 生成报警信息
      alarmUtils.generateAlarmInfo(name, buf.alarmInfo, "数量达标",
                                   config.get());
      // 生成报警图片并画框
      alarmUtils.saveAlarmImage(buf.alarmInfo.alarmFile + "/" +
                                    buf.alarmInfo.alarmId + ".jpg",
                                *image, buf.frameType, config->isDraw, rbboxes);
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
    }
  }
}

FlowEngineModuleRegister(ObjectNumberModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
