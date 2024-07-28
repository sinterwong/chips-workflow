/**
 * @file objectCounterModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-04-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "objectCounterModule.hpp"
#include "logger/logger.hpp"

#include <cassert>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace module {

/**
 * @brief
 *
 * @param message
 */
void ObjectCounterModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} ObjectCounterModule was done!", name);
      stopFlag.store(true);
      return;
    }

    // 读取图片
    frame_ptr frameBuf = ptr->pools->read(buf.steramName, buf.key);
    if (!frameBuf) {
      FLOWENGINE_LOGGER_WARN("{} ObjectCounterModule read frame is failed!",
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

    assert(regions.size() == 1);

    // 根据提供的配置执行算法，
    auto &detNet = config->algoPipelines.at(0);
    auto &attentions = detNet.second.attentions; // 检测关注的类别
    auto &reidNet = config->algoPipelines.at(1);
    InferParams detParams{name,
                          buf.frameType,
                          detNet.second.cropScaling,
                          regions.at(0),
                          {image->cols, image->rows, image->channels()}};

    InferResult detRet;
    if (!ptr->algo->infer(detNet.first, image->data, detParams, detRet)) {
      return;
    };
    // 获取检测结果信息
    auto bboxes = std::get_if<common::BBoxes>(&detRet.aRet);
    if (!bboxes) {
      FLOWENGINE_LOGGER_ERROR("ObjectCounterModule: Wrong algorithm type!");
      return;
    }

    std::vector<std::pair<int, infer::solution::DETECTBOX>> results;

    infer::solution::DETECTIONS detections;
    for (auto &bbox : *bboxes) {
      // 需要过滤掉不关注的类别
      if (!attentions.empty()) {
        auto iter = std::find(attentions.begin(), attentions.end(),
                              static_cast<int>(bbox.class_id));
        if (iter == attentions.end()) { // 该类别没有出现在关注类别中
          continue;
        }
      }
      infer::solution::DETECTION_ROW tmpRow;
      tmpRow.tlwh = infer::solution::DETECTBOX{bbox.bbox[0], bbox.bbox[1],
                                               bbox.bbox[2] - bbox.bbox[0],
                                               bbox.bbox[3] - bbox.bbox[1]};
      tmpRow.confidence = bbox.class_confidence;
      InferParams reidParams{
          name,
          buf.frameType,
          reidNet.second.cropScaling,
          common::RetBox{detNet.first, static_cast<int>(bbox.bbox[0]),
                         static_cast<int>(bbox.bbox[1]),
                         static_cast<int>(bbox.bbox[2] - bbox.bbox[0]),
                         static_cast<int>(bbox.bbox[3] - bbox.bbox[1]),
                         bbox.class_confidence,
                         static_cast<int>(bbox.class_id)},
          {image->cols, image->rows, image->channels()}};
      InferResult reidRet;
      if (!ptr->algo->infer(reidNet.first, image->data, reidParams, reidRet)) {
        return;
      };
      auto feature = std::get_if<common::Eigenvector>(&reidRet.aRet);
      if (!feature) {
        FLOWENGINE_LOGGER_ERROR("Feature extract is failed!");
        continue;
      }
      for (size_t i = 0; i < feature->size(); ++i) {
        tmpRow.feature[i] = feature->at(i);
      }
      detections.emplace_back(tmpRow);
    }

    deepsort->predict();
    deepsort->update(detections); // TODO 这里没有考虑抽帧的情况

    for (infer::solution::Track &track : deepsort->tracks) {
      if (!track.is_confirmed() || track.time_since_update > 1)
        continue;
      results.push_back(std::make_pair(track.track_id, track.to_tlwh()));
    }

    for (size_t j = 0; j < results.size(); j++) {
      counter.insert(results[j].first);
    }

    FLOWENGINE_LOGGER_DEBUG("person number: {}", counter.size());

    // 每到一定的数量就会触发报警
    if (counter.size() != 0 &&
        counter.size() % static_cast<size_t>(config->amount) == 0) {
      // 生成报警信息
      alarmUtils.generateAlarmInfo(name, buf.alarmInfo, "达到计数要求",
                                   config.get());
      // 生成报警图片，此处相当于截了个图
      alarmUtils.saveAlarmImage(
          buf.alarmInfo.alarmFile + "/" + buf.alarmInfo.alarmId + ".jpg",
          *image, buf.frameType, static_cast<DRAW_TYPE>(config->drawType));
      autoSend(buf);
      counter.insert(counter.size() * 200);
      break;
    }
  }
}

} // namespace module
