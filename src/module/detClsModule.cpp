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
#include <cstddef>
#include <cstdlib>
#include <experimental/filesystem>

#include "infer/preprocess.hpp"
#include "logger/logger.hpp"
#include "module_utils.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <unordered_map>

namespace module {

/**
 * @brief
 *
 * @param message
 */
void DetClsModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} HelmetModule module was done!", name);
      stopFlag.store(true);
      return;
    }

    FLOWENGINE_LOGGER_INFO("DetClsModule is liveliness");

    // 读取图片
    FrameBuf frameBufMessage = ptr->pool->read(buf.key);
    auto image =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

    // 各个算法结果的区域
    std::unordered_map<std::string, std::vector<common::RetBox>> algoRegions;

    // 初始待计算区域，每次算法结果出来之后需要更新regions
    std::vector<common::RetBox> regions;
    for (auto const &area : config->regions) {
      regions.emplace_back(common::RetBox{
          name,
          {static_cast<float>(area[0][0]), static_cast<float>(area[0][1]),
           static_cast<float>(area[1][0]), static_cast<float>(area[1][1]), 0.0,
           0.0}});
    }
    algoRegions["regions"] = std::move(regions);

    FLOWENGINE_LOGGER_CRITICAL("hello");

    // 根据提供的配置执行算法，
    auto &apipes = config->algoPipelines;
    for (auto const &ap : apipes) {
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
                                  {region.second[0] + bbox.bbox[0],
                                   region.second[1] + bbox.bbox[1],
                                   region.second[2] + bbox.bbox[0],
                                   region.second[3] + bbox.bbox[1],
                                   bbox.det_confidence, bbox.class_id}};
              algoRegions[ap.first].emplace_back(std::move(b));
            }
            break;
          }
          default: {
            FLOWENGINE_LOGGER_ERROR("Nonsupportd algo type");
            break;
          }
          }
        }
      }
    }

    // 至此，所有的算法模块执行完成，整合算法结果判断是否报警
    auto lastPipeName = apipes.at(apipes.size() - 1).first;
    cv::Mat showImage = image->clone();
    for (auto const& alarmBox : algoRegions.at(lastPipeName)) {
      utils::drawRetBox(showImage, alarmBox);
    }
    cv::imwrite("temp.jpg", showImage);
    autoSend(buf);
  }
}

FlowEngineModuleRegister(DetClsModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
