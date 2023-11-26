/**
 * @file detClsModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-09-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "detClsModule.h"
#include "logger/logger.hpp"
#include "video_utils.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace module {

using common::RetBox;

void DetClsModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} DetClsModule was done!", name);
      stopFlag.store(true);
      return;
    }

    // 读取图片
    frame_ptr frameBuf = ptr->pools->read(buf.steramName, buf.key);
    if (!frameBuf) {
      FLOWENGINE_LOGGER_WARN("{} DetClsModule read frame is failed!", name);
      return;
    }
    auto image = std::any_cast<std::shared_ptr<cv::Mat>>(frameBuf->read("Mat"));

    // 各个算法结果的区域
    std::unordered_map<std::string, std::vector<common::RetBox>> algoRegions;

    // 初始待计算区域，每次算法结果出来之后需要更新regions
    std::vector<common::RetBox> regions;
    for (auto const &area : config->regions) {
      regions.emplace_back(RetBox(name, area));
    }
    if (regions.empty()) {
      // 前端没有画框
      regions.emplace_back(RetBox{name});
    }
    algoRegions["regions"] = std::move(regions);

    // 根据提供的配置执行算法，
    auto &apipes = config->algoPipelines;
    for (auto const &ap : apipes) {
      algoRegions[ap.first] = std::vector<common::RetBox>();
      auto &attentions = ap.second.attentions; // 哪些类别将被保留
      auto &basedNames = ap.second.basedNames; // 基于哪些算法的结果去执行
      for (auto const &bn : basedNames) {
        auto &regions = algoRegions.at(bn);
        for (auto const &region : regions) {
          InferParams params{name,
                             buf.frameType,
                             ap.second.cropScaling,
                             region,
                             {image->cols, image->rows, image->channels()}};
          InferResult ret;
          if (!ptr->algo->infer(ap.first, image->data, params, ret)) {
            return;
          };
          auto atype = ptr->algo->getType(ap.first);
          // 将结果计入algo regions，便于后续其他算法使用
          switch (atype) {
          case common::AlgoRetType::Classifier: {
            auto cls = std::get_if<common::ClsRet>(&ret.aRet);
            if (!cls)
              continue;

            // 需要过滤掉不关注的类别
            if (!attentions.empty()) {
              auto iter = std::find(attentions.begin(), attentions.end(),
                                    static_cast<int>(cls->first));
              if (iter == attentions.end()) { // 该类别没有出现在关注类别中
                continue;
              }
            }
            RetBox b{ap.first,      region.x,    region.y,  region.width,
                     region.height, cls->second, cls->first};
            algoRegions[ap.first].emplace_back(std::move(b));
            break;
          }
          case common::AlgoRetType::Detection: {
            auto bboxes = std::get_if<common::BBoxes>(&ret.aRet);
            if (!bboxes)
              continue;
            for (auto &bbox : *bboxes) {
              // 需要过滤掉不关注的类别
              if (!attentions.empty()) {
                auto iter = std::find(attentions.begin(), attentions.end(),
                                      static_cast<int>(bbox.class_id));
                if (iter == attentions.end()) { // 该类别没有出现在关注类别中
                  continue;
                }
              }
              // 记录算法结果
              common::RetBox b{ap.first,
                               static_cast<int>(bbox.bbox[0] + region.x),
                               static_cast<int>(bbox.bbox[1] + region.y),
                               static_cast<int>(bbox.bbox[2] - bbox.bbox[0]),
                               static_cast<int>(bbox.bbox[3] - bbox.bbox[1]),
                               bbox.det_confidence,
                               static_cast<int>(bbox.class_id)};
              algoRegions[ap.first].emplace_back(std::move(b));
            }
            break;
          }
          default: {
            FLOWENGINE_LOGGER_ERROR("{}: Nonsupportd algo type", ap.first);
            break;
          }
          }
        }
      }
    }

    // 至此，所有的算法模块执行完成，整合算法结果判断是否报警
    auto lastPipeName = apipes.at(apipes.size() - 1).first;
    auto &alarmRegions = algoRegions.at(lastPipeName);
    RetBox alarmBox = {name};
    bool isAlarm = false;
    if ((alarmRegions.size() > 0) == (config->requireExistence)) {
      /**
       * @brief 报警条件：关注类别是否存在 == 存在关注类别后报/不报警
       *  设关注类型是否存在为a, 存在关注类别后报/不报警为b:
       * 会报警的情况：a=true and b=true, a=false and b=false
       * 当 a=true，b=true, 此时需要通过阈值判定，当阈值通过才会报警
       */
      if (alarmRegions.size() == 0) {
        isAlarm = true; // a=false and b=false的情况
      } else {
        // a=true and b=true的情况
        for (auto const &box : alarmRegions) {
          if (box.confidence > config->threshold) {
            // 存在符合条件的报警
            alarmBox = box;
            isAlarm = true;
            FLOWENGINE_LOGGER_DEBUG("{}: {}, {}, {}!", name, alarmBox.name,
                                    alarmBox.confidence, alarmBox.idx);
            break;
          }
        }
      }
    }
    if (isAlarm) { // 需要报警
      // 生成报警信息
      alarmUtils.generateAlarmInfo(name, buf.alarmInfo, "存在报警行为",
                                   config.get());
      // 生成报警图片，带框和不带框共两张
      alarmUtils.saveAlarmImage(
          buf.alarmInfo.alarmFile + "/" + buf.alarmInfo.alarmId + ".jpg",
          *image, buf.frameType, static_cast<DRAW_TYPE>(2), {alarmBox});

      // 本轮算法结果生成
      json algoRet;
      for (auto &info : algoRegions) {
        std::string bboxesJson, aname;
        utils::retBoxes2json(info.second, bboxesJson);
        algoRet[info.first] = std::move(bboxesJson);
      }
      buf.alarmInfo.algorithmResult = algoRet.dump();
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

FlowEngineModuleRegister(DetClsModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
