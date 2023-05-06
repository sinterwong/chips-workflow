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

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "messageBus.h"
#include "module.hpp"
#include "module_utils.hpp"
#include "preprocess.hpp"

#include "videoRecord.hpp"

using common::AlarmBase;

using namespace video;

namespace module {
namespace filesystem = std::experimental::filesystem;
class AlarmUtils {
protected:
  bool isRecord = false; // 是否是保存视频状态
  int frameCount = 0;    // 保存帧数
  int drawTimes = 0;     // 视频上画的次数
  std::unique_ptr<VideoRecord> vr;

public:
  AlarmUtils() {}
  virtual ~AlarmUtils() { destoryOutputStream(); }

  inline void destoryOutputStream() {
    if (isRecording()) {
      vr->destory();
    }
  }

  inline bool initRecorder(std::string const &path, int width, int height,
                           int rate, int videDuration) {
    isRecord = true;
    frameCount = videDuration * rate;
    drawTimes = floor(frameCount / 3);
    videoOptions params;
    params.resource = path;
    params.width = width;
    params.height = height;
    params.frameRate = rate;
    try {
      vr = std::make_unique<VideoRecord>(std::move(params));
    } catch (const std::runtime_error &e) {
      FLOWENGINE_LOGGER_ERROR("initRecorder exception: ", e.what());
      return false;
    }
    return vr->init();
  }

  inline void recordVideo(cv::Mat &frame, RetBox const &bbox) {
    if (drawTimes-- > 0) {
      utils::drawRetBox(frame, bbox, cv::Scalar{255, 0, 0});
    }
    vr->record(frame.data);
    if (!vr->check() || --frameCount <= 0) {
      isRecord = false;
      frameCount = 0;
      drawTimes = 0;
      vr->destory();
    }
  }

  inline bool isRecording() { return isRecord; }

  inline void generateAlarmInfo(std::string const &name, AlarmInfo &alarmInfo,
                                std::string const &detail,
                                AlarmBase const *const config) {
    // 生成本次报警的唯一ID
    alarmInfo.alarmId = utils::generate_hex(16);
    alarmInfo.alarmFile = config->outputDir + "/" + alarmInfo.alarmId;
    alarmInfo.prepareDelayInSec = config->videoDuration;
    alarmInfo.alarmDetails = detail;
    alarmInfo.alarmType = name;
    alarmInfo.page = config->page;
    alarmInfo.eventId = config->eventId;

    // 生成报警文件夹
    filesystem::create_directories(alarmInfo.alarmFile);
  }

  inline void saveAlarmImage(std::string const &path, cv::Mat const &frame,
                             ColorType const ctype, bool isDraw = false,
                             RetBox bbox = RetBox{"", {0, 0, 0, 0, 0, 0}}) {
    cv::Mat showImage;
    // 临时画个图（后续根据前端参数来决定返回的图片是否带有画图标记）
    switch (ctype) {
    case ColorType::RGB888: {
      showImage = frame.clone();
      cv::cvtColor(showImage, showImage, cv::COLOR_RGB2BGR);
      break;
    }
    case ColorType::NV12: {
      showImage = frame.clone();
      cv::cvtColor(showImage, showImage, cv::COLOR_YUV2BGR_NV12);
      break;
    }
    case ColorType::BGR888:
    case ColorType::None: {
      showImage = frame;
      break;
    }
    }
    // 画报警框
    if (isDraw) {
      utils::drawRetBox(showImage, bbox);
    }

    // // 报警框名称操作
    // auto pos = name.find("_");
    // alarmBox.first = name.substr(pos + 1).substr(0, name.substr(pos +
    // 1).find("_"));

    // 输出alarm image
    cv::imwrite(path, showImage);
  }
};
} // namespace module
#endif // __METAENGINE_LOGIC_MODULE_H_
