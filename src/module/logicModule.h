/**
 * @file logicModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-10
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_LOGIC_MODULE_H_
#define __METAENGINE_LOGIC_MODULE_H_

#include <experimental/filesystem>
#include <memory>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "common/config.hpp"
#include "preprocess.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include "module_utils.hpp"

#if (TARGET_PLATFORM == 0)
#include "x3/videoRecord.hpp"
using namespace module::utils;
#elif (TARGET_PLATFORM == 1)
#include "jetson/videoRecord.hpp"
#elif (TARGET_PLATFORM == 2)
#include "jetson/trt_inference.hpp"
#endif

namespace module {
namespace filesystem = std::experimental::filesystem;
class LogicModule : public Module {
protected:
  bool isRecord = false;      // 是否是保存视频状态
  int frameCount = 0;         // 保存帧数
  int drawTimes = 0;          // 视频上画的次数
  common::LogicConfig params; // 逻辑参数
  retBox alarmBox;            // 报警作图框
  std::unique_ptr<utils::VideoRecord> vr;

public:
  LogicModule(Backend *ptr, const std::string &initName,
              const std::string &initType, const common::LogicConfig &params_)
      : Module(ptr, initName, initType), params(params_) {}
  virtual ~LogicModule() {}

  inline bool drawBox(cv::Mat &image, retBox const &bbox,
                      cv::Scalar const &scalar = {0, 0, 255}) const {
    cv::Rect rect(bbox.second[0], bbox.second[1],
                  bbox.second[2] - bbox.second[0],
                  bbox.second[3] - bbox.second[1]);
    cv::rectangle(image, rect, scalar, 2);
    cv::putText(image, bbox.first, cv::Point(rect.x, rect.y - 1),
                cv::FONT_HERSHEY_PLAIN, 2, {255, 255, 255});
    return true;
  }

  inline bool drawResult(cv::Mat &image, AlgorithmResult const &rm) const {
    for (auto &bbox : rm.bboxes) {
      drawBox(image, bbox,
              cv::Scalar{static_cast<double>(rand() % 255),
                         static_cast<double>(rand() % 255),
                         static_cast<double>(rand() % 255)});
    }

    for (auto &poly : rm.polys) {
      std::vector<cv::Point> fillContSingle;
      for (int i = 0; i < static_cast<int>(poly.second.size()); i += 2) {
        fillContSingle.emplace_back(
            cv::Point{static_cast<int>(poly.second[i]),
                      static_cast<int>(poly.second[i + 1])});
      }
      cv::fillPoly(image, std::vector<std::vector<cv::Point>>{fillContSingle},
                   cv::Scalar(0, 255, 255));
    }

    return true;
  }

  inline void destoryOutputStream() { vr->destory(); }

  inline bool initRecord(queueMessage const &buf) {
    isRecord = true;
    int frameRate =
        buf.cameraResult.frameRate > 0 ? buf.cameraResult.frameRate : 30;
    frameCount = params.videDuration * frameRate;
    drawTimes = floor(frameCount / 3);
    std::string filepath =
        buf.alarmResult.alarmFile + "/" + buf.alarmResult.alarmId + ".mp4";
    videoOptions params;
    params.resource = std::move(filepath);
    params.width = buf.cameraResult.widthPixel;
    params.height = buf.cameraResult.heightPixel;
    params.frameRate = frameRate;
    vr = std::make_unique<utils::VideoRecord>(std::move(params));
    return vr->init();
  }

  inline void recordVideo(int key, int width, int height) {

    FrameBuf frameBufMessage = backendPtr->pool->read(key);
    if (drawTimes-- > 0) {
      auto image =
          std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
      drawBox(*image, alarmBox, cv::Scalar{255, 0, 0});
      vr->record(image->data);
    } else {
      vr->record(std::any_cast<void *>(frameBufMessage.read("void**")));
    }

    if (--frameCount <= 0 || !vr->check()) {
      isRecord = false;
      frameCount = 0;
      drawTimes = 0;
      vr->destory();
    }
  }

  inline void generateAlarm(queueMessage &buf, std::string const &detail,
                            retBox const &bbox) {
    // 生成本次报警的唯一ID
    buf.alarmResult.alarmVideoDuration = params.videDuration;
    buf.alarmResult.alarmId = utils::generate_hex(16);
    buf.alarmResult.alarmFile =
        params.outputDir + "/" + buf.alarmResult.alarmId;
    buf.alarmResult.alarmDetails = detail;
    buf.alarmResult.alarmType = name;
    buf.alarmResult.page = params.page;
    buf.alarmResult.eventId = params.eventId;

    // TODO
    cv::Mat showImage;
    FrameBuf frameBufMessage = backendPtr->pool->read(buf.key);
    auto frame =
        std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
    if (params.isDraw) {
      // 临时画个图（后续根据前端参数来决定返回的图片是否带有画图标记）
      showImage = frame->clone();
      switch (buf.frameType) {
      case ColorType::BGR888:
        break;
      case ColorType::RGB888: {
        cv::cvtColor(showImage, showImage, cv::COLOR_RGB2BGR);
        break;
      }
      case ColorType::NV12: {
        cv::cvtColor(showImage, showImage, cv::COLOR_YUV2BGR_NV12);
        break;
      }
      case ColorType::None:
        break;
      }
    } else {
      showImage = *frame;
    }

    // 记录当前的框为报警框
    alarmBox = bbox;
    auto pos = name.find("_");
    alarmBox.first =
        name.substr(pos + 1).substr(0, name.substr(pos + 1).find("_"));
    // TODO 报警框放大

    // 单独画出报警框
    drawBox(showImage, alarmBox);
    // 画出所有结果
    // drawResult(showImage, buf.algorithmResult);
    buf.algorithmResult.bboxes.push_back(alarmBox);
    filesystem::create_directories(buf.alarmResult.alarmFile);
    std::string imagePath =
        buf.alarmResult.alarmFile + "/" + buf.alarmResult.alarmId + ".jpg";
    // 输出alarm image
    cv::imwrite(imagePath, showImage);
  }
};
} // namespace module
#endif // __METAENGINE_LOGIC_MODULE_H_
