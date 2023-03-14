/**
 * @file helmetModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-09-01
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "generalModule.h"
#include <cstddef>
#include <cstdlib>
#include <experimental/filesystem>

#include "infer/preprocess.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace module {

/**
 * @brief
 *
 * @param message
 */
void GeneralModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    if (type == MessageType::Close) {
      FLOWENGINE_LOGGER_INFO("{} HelmetModule module was done!", name);
      stopFlag.store(true);
      return;
    }

    FrameBuf frameBufMessage = ptr->pool->read(buf.key);
    auto image =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));

    cv::Rect region = cv::Rect{config->region[0][0], config->region[0][1],
                               config->region[1][0], config->region[1][1]};

    cv::Mat inferImage;
    if (region.area() > 1) {

      if (!infer::utils::cropImage(*image, inferImage, region, buf.frameType)) {
        FLOWENGINE_LOGGER_ERROR("cropImage is failed, rect is {},{},{},{}, "
                                "but the video resolution is {}x{}",
                                region.x, region.y, region.width, region.height,
                                image->cols, image->rows);
        return;
      } else {
        inferImage = *image; // 需不需要clone再说吧
      }
    }

    // 此处开始逻辑编写
    auto &apipes = config->algoPipelines;
    // 按顺序执行每一个算法，后面的算法基于前面的算法
    for (auto const &ap : apipes) {
      std::vector<common::RetBox> regions{
          {name, {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};
      InferParams params{
          name,
          buf.frameType,
          0.5,
          regions,
          {inferImage.cols, inferImage.rows, inferImage.channels()}};
      InferResult ret;
      std::string aname = ap.first;
    }
  }
}

FlowEngineModuleRegister(GeneralModule, backend_ptr, std::string const &,
                         MessageType const &, ModuleConfig &);
} // namespace module
