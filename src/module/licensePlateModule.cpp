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
#include "infer/postprocess.hpp"
#include "infer/preprocess.hpp"
#include "logger/logger.hpp"
#include "module_utils.hpp"

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <thread>
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
          {static_cast<float>(area[0].x), static_cast<float>(area[0].y),
           static_cast<float>(area[1].x), static_cast<float>(area[1].y), 0.0,
           0.0}});
    }
    if (regions.empty()) {
      // 前端没有画框
      regions.emplace_back(common::RetBox{name, {0, 0, 0, 0, 0, 0}});
    }

    assert(regions.size() == 1);

    auto &lpDet = config->algoPipelines.at(0);
    auto &lpRec = config->algoPipelines.at(1);

    // 至此，所有的算法模块执行完成，整合算法结果判断是否报警
    InferParams detParams{name,
                          buf.frameType,
                          lpDet.second.cropScaling,
                          regions.at(0),
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
    std::vector<OCRRet> results;
    for (auto &kbbox : *kbboxes) {
      cv::Mat licensePlateImage;
      cv::Rect2i rect{
          static_cast<int>(kbbox.bbox.bbox[0]),
          static_cast<int>(kbbox.bbox.bbox[1]),
          static_cast<int>(kbbox.bbox.bbox[2] - kbbox.bbox.bbox[0]),
          static_cast<int>(kbbox.bbox.bbox[3] - kbbox.bbox.bbox[1])};
      infer::utils::cropImage(*image, licensePlateImage, rect, buf.frameType);
      if (kbbox.bbox.class_id == 1) {
        // 车牌矫正
        infer::utils::sortFourPoints(kbbox.points); // 排序关键点
        cv::Mat lpr_rgb;
        cv::cvtColor(licensePlateImage, lpr_rgb, cv::COLOR_YUV2RGB_NV12);
        cv::Mat lpr_ted;
        // 相对于车牌图片的点
        infer::utils::fourPointTransform(lpr_rgb, lpr_ted, kbbox.points);

        // 上下分割，垂直合并车牌
        splitMerge(lpr_ted, licensePlateImage);
        infer::utils::RGB2NV12(licensePlateImage, licensePlateImage);
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
        FLOWENGINE_LOGGER_ERROR("LicensePlateModule: Wrong algorithm type!");
        return;
      }
      if (charIds->size() != 7 && charIds->size() != 8) {
        continue; // 过滤无效的车牌
      }
      auto lpr = getChars(*charIds, charsetsMapping);
      FLOWENGINE_LOGGER_CRITICAL("License plate number is: {}", lpr);
      results.emplace_back(OCRRet{kbbox, *charIds, std::move(lpr)});
    }

    if (!results.empty()) {
      // 本轮算法结果生成
      utils::retOCR2json(results, buf.alarmInfo.algorithmResult);
      alarmUtils.generateAlarmInfo(name, buf.alarmInfo, "存在车牌",
                                   config.get());
      autoSend(buf);
    }
  }
  std::this_thread::sleep_for(std::chrono::microseconds{config->interval});
}

FlowEngineModuleRegister(LicensePlateModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
