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

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace module {

/**
 * @brief
 *
 * @param message
 */
void DetClsModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} DetClsModule was done!", name);
      stopFlag.store(true);
      return;
    }

    // 读取图片
    FrameBuf frameBufMessage = ptr->pool->read(buf.key);
    auto image =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

    if (alarmUtils.isRecording()) {
      alarmUtils.recordVideo(*image, alarmBox);
      break;
    }

    // 各个算法结果的区域
    std::unordered_map<std::string, std::vector<common::RetBox>> algoRegions;

    // 初始待计算区域，每次算法结果出来之后需要更新regions
    std::vector<common::RetBox> regions;
    for (auto const &area : config->regions) {
      regions.emplace_back(common::RetBox{
          name,
          {static_cast<float>(area[0].x), static_cast<float>(area[0].y),
           static_cast<float>(area[1].x), static_cast<float>(area[1].y), 0.0,
           0.0}});
    }
    if (regions.empty()) {
      // 前端没有画框
      regions.emplace_back(common::RetBox{name, {0, 0, 0, 0, 0, 0}});
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
            common::RetBox b = {ap.first,
                                {region.second[0], region.second[1],
                                 region.second[2], region.second[3],
                                 cls->second, static_cast<float>(cls->first)}};
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
              common::RetBox b = {ap.first,
                                  {bbox.bbox[0] + region.second[0],
                                   bbox.bbox[1] + region.second[1],
                                   bbox.bbox[2] + region.second[0],
                                   bbox.bbox[3] + region.second[1],
                                   bbox.det_confidence, bbox.class_id}};
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
    if (alarmRegions.size() > 0) {
      for (auto const &box : algoRegions.at(lastPipeName)) {
        alarmBox = box;
        if (alarmBox.second[4] > config->threshold) {
          // 存在符合条件的报警
          FLOWENGINE_LOGGER_DEBUG("{}: {}, {}, {}!", name, alarmBox.first,
                                     alarmBox.second[4], alarmBox.second[5]);
          // 生成报警信息
          alarmUtils.generateAlarmInfo(name, buf.alarmInfo, "存在报警行为",
                                       config.get());
          // 生成报警图片
          alarmUtils.saveAlarmImage(
              buf.alarmInfo.alarmFile + "/" + buf.alarmInfo.alarmId + ".jpg",
              *image, buf.frameType, config->isDraw, alarmBox);

          // 初始化报警视频
          alarmUtils.initRecorder(buf.alarmInfo.alarmFile + "/" +
                                      buf.alarmInfo.alarmId + ".mp4",
                                  buf.alarmInfo.width, buf.alarmInfo.height, 25,
                                  config->videoDuration);

          // 本轮算法结果生成
          json algoRet;
          for (auto &info : algoRegions) {
            std::string bboxesJson, aname;
            utils::retBoxes2json(info.second, bboxesJson);

            // // 模块名称提取
            // std::string_view str_view = info.first;
            // auto pos1 = str_view.find('_');
            // auto pos2 = str_view.find('_', pos1 + 1);
            // aname = str_view.substr(pos1 + 1, pos2 - pos1 - 1);
            algoRet[info.first] = std::move(bboxesJson);
          }
          buf.alarmInfo.algorithmResult = algoRet.dump();
          autoSend(buf);
          break;
        }
      }
    }
  }
  if (!alarmUtils.isRecording()) {
    std::this_thread::sleep_for(std::chrono::microseconds{config->interval});
  }
}

FlowEngineModuleRegister(DetClsModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
