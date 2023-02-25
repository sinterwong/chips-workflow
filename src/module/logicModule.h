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
#include <stdexcept>
#include <vector>

#include "common/config.hpp"
#include "logger/logger.hpp"
#include "messageBus.h"
#include "module.hpp"
#include "module_utils.hpp"
#include "preprocess.hpp"

#include "videoRecord.hpp"

using common::LogicConfig;

namespace module {
namespace filesystem = std::experimental::filesystem;
class LogicModule : public Module {
protected:
  bool isRecord = false; // 是否是保存视频状态
  int frameCount = 0;    // 保存帧数
  int drawTimes = 0;     // 视频上画的次数
  LogicConfig config;    // 逻辑参数
  std::unique_ptr<utils::VideoRecord> vr;

public:
  LogicModule(Backend *ptr, const std::string &initName,
              const std::string &initType, const LogicConfig &config_)
      : Module(ptr, initName, initType), config(config_) {}
  virtual ~LogicModule() {}

  inline void destoryOutputStream() { vr->destory(); }

  inline bool initRecorder(std::string const &path, int width, int height,
                           int rate) {
    isRecord = true;
    frameCount = config.videDuration * rate;
    drawTimes = floor(frameCount / 3);
    videoOptions params;
    params.resource = path;
    params.width = width;
    params.height = height;
    params.frameRate = rate;
    try {
      vr = std::make_unique<utils::VideoRecord>(std::move(params));
    } catch (const std::runtime_error &e) {
      FLOWENGINE_LOGGER_ERROR("initRecorder exception: ", e.what());
      return false;
    }
    return vr->init();
  }

  inline void recordVideo(int key, int width, int height, RetBox const &bbox) {

    FrameBuf fbm = backendPtr->pool->read(key);
    auto image = std::any_cast<std::shared_ptr<cv::Mat>>(fbm.read("Mat"));
    if (drawTimes-- > 0) {
      utils::drawRetBox(*image, bbox, cv::Scalar{255, 0, 0});
    }
    vr->record(image->data);
    if (!vr->check() || --frameCount <= 0) {
      isRecord = false;
      frameCount = 0;
      drawTimes = 0;
      vr->destory();
    }
  }

  inline void generateAlarmInfo(AlarmInfo &alarmInfo, std::string const &detail,
                                RetBox const &bbox) {
    // 生成本次报警的唯一ID
    alarmInfo.alarmId = utils::generate_hex(16);
    alarmInfo.alarmFile = config.outputDir + "/" + alarmInfo.alarmId;
    alarmInfo.prepareDelayInSec = config.videDuration;
    alarmInfo.alarmDetails = detail;
    alarmInfo.alarmType = name;
    alarmInfo.page = config.page;
    alarmInfo.eventId = config.eventId;

    // 生成报警文件夹
    filesystem::create_directories(alarmInfo.alarmFile);
  }

  inline void saveAlarmImage(std::string const &path, cv::Mat const &frame,
                             ColorType const ctype, RetBox const &bbox) {
    cv::Mat showImage;
    if (config.isDraw) {
      // 临时画个图（后续根据前端参数来决定返回的图片是否带有画图标记）
      showImage = frame.clone();
      switch (ctype) {
      case ColorType::RGB888: {
        cv::cvtColor(showImage, showImage, cv::COLOR_RGB2BGR);
        break;
      }
      case ColorType::NV12: {
        cv::cvtColor(showImage, showImage, cv::COLOR_YUV2BGR_NV12);
        break;
      }
      case ColorType::BGR888:
      case ColorType::None:
        break;
      }
    } else {
      showImage = frame;
    }

    // // 报警框名称操作
    // auto pos = name.find("_");
    // alarmBox.first = name.substr(pos + 1).substr(0, name.substr(pos +
    // 1).find("_"));

    // 画报警框
    utils::drawRetBox(showImage, bbox);

    // 输出alarm image
    cv::imwrite(path, showImage);
  }
};
} // namespace module
#endif // __METAENGINE_LOGIC_MODULE_H_
