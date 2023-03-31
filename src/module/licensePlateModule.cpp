/**
 * @file licensePlateModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-30
 *
 * @copyright Copyright (c) 2023
 *
 */
#include "licensePlateModule.h"
#include "infer/preprocess.hpp"
#include "logger/logger.hpp"
#include "module_utils.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <variant>

using json = nlohmann::json;
using common::KeypointsBoxes;
using common::OCRRet;

namespace module {

/**
 * @brief
 *
 * @param message
 */
void LicensePlateModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} HelmetModule module was done!", name);
      stopFlag.store(true);
      return;
    }

    // 读取图片
    FrameBuf frameBufMessage = ptr->pool->read(buf.key);
    auto image =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

    // 初始待计算区域，每次算法结果出来之后需要更新regions
    std::vector<common::RetBox> regions;
    for (auto const &area : config->regions) {
      regions.emplace_back(common::RetBox{
          name,
          {static_cast<float>(area[0][0]), static_cast<float>(area[0][1]),
           static_cast<float>(area[1][0]), static_cast<float>(area[1][1]), 0.0,
           0.0}});
    }
    if (regions.empty()) {
      // 前端没有画框
      regions.emplace_back(common::RetBox{name, {0, 0, 0, 0, 0, 0}});
    }

    auto &lpDet = config->algoPipelines.at(0);
    auto &lpRec = config->algoPipelines.at(1);

    std::vector<OCRRet> results;

    // 至此，所有的算法模块执行完成，整合算法结果判断是否报警
    for (auto &region : regions) {
      InferParams detParams{name,
                            buf.frameType,
                            lpDet.second.cropScaling,
                            region,
                            {image->cols, image->rows, image->channels()}};
      InferResult textDetRet;
      if (!ptr->algo->infer(lpDet.first, image->data, detParams, textDetRet)) {
        return;
      };

      // 获取到文本检测的信息
      auto kbboxes = std::get_if<KeypointsBoxes>(&textDetRet.aRet);
      if (!kbboxes) {
        FLOWENGINE_LOGGER_ERROR("OCRModule: Wrong algorithm type!");
        return;
      }

      // 识别每一个车牌
      for (auto &kbbox : *kbboxes) {
        cv::Mat licensePlateImage;
        cv::Rect2i rect{
            static_cast<int>(kbbox.bbox.bbox[0]),
            static_cast<int>(kbbox.bbox.bbox[1]),
            static_cast<int>(kbbox.bbox.bbox[2] - kbbox.bbox.bbox[0]),
            static_cast<int>(kbbox.bbox.bbox[3] - kbbox.bbox.bbox[1])};
        infer::utils::cropImage(*image, licensePlateImage, rect, buf.frameType);
        if (kbbox.bbox.class_id == 1) {
          // 如果是双层车牌
          // TODO 矫正
          // 上下分割，垂直合并车牌
          splitMerge(licensePlateImage, licensePlateImage);
        }
        InferParams recParams{name,
                              buf.frameType,
                              0.0,
                              common::RetBox{name, {0, 0, 0, 0, 0, 0}},
                              {licensePlateImage.cols, licensePlateImage.rows,
                               licensePlateImage.channels()}};
        InferResult textRecRet;
        if (!ptr->algo->infer(lpRec.first, licensePlateImage.data, recParams,
                              textRecRet)) {
          return;
        };
        auto charIds = std::get_if<CharsRet>(&textRecRet.aRet);
        if (!charIds) {
          FLOWENGINE_LOGGER_ERROR("OCRModule: Wrong algorithm type!");
          return;
        }
        results.emplace_back(OCRRet{kbbox, *charIds, getChars(*charIds)});
      }
    }

    // 本轮算法结果生成
    utils::retOCR2json(results, buf.alarmInfo.algorithmResult);
    autoSend(buf);
  }
}

FlowEngineModuleRegister(LicensePlateModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
