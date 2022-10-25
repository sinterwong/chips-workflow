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

#include <any>
#include <experimental/filesystem>
#include <memory>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <sstream>
#include <type_traits>
#include <vector>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "module.hpp"
#include "utils/convertMat.hpp"
// #include "videoOutput.h"

namespace module {
class LogicModule : public Module {
protected:
  bool isRecord = false;                     // 是否是保存视频状态
  int frameCount = 0;                        // 保存帧数
  int drawTimes = 0;                         // 视频上画的次数
  // std::unique_ptr<videoOutput> outputStream; // 输出流
  common::LogicConfig params;                // 逻辑参数
  retBox alarmBox;                           // 报警作图框
  // utils::ImageConverter imageConverter; // mat to base64

  inline unsigned int random_char() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    return dis(gen);
  }

  inline std::string generate_hex(const unsigned int len) {
    std::stringstream ss;
    for (auto i = 0; i < static_cast<int>(len); i++) {
      const auto rc = random_char();
      std::stringstream hexstream;
      hexstream << std::hex << rc;
      auto hex = hexstream.str();
      ss << (hex.length() < 2 ? '0' + hex : hex);
    }
    return ss.str();
  }

public:
  LogicModule(Backend *ptr, const std::string &initName,
              const std::string &initType, const common::LogicConfig &params_,
              const std::vector<std::string> &recv = {},
              const std::vector<std::string> &send = {})
      : Module(ptr, initName, initType, recv, send), params(params_) {}
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

  inline void bboxScaling(retBox const &bbox) {}

  inline void checkOutputStream() {
    // if (outputStream && outputStream->IsStreaming()) {
    //     outputStream->Close();
    //   };
  }

  inline void initRecord(queueMessage const &buf) {
    // videoOptions opt;
    // opt.resource =
    //     buf.alarmResult.alarmFile + "/" + buf.alarmResult.alarmId + ".mp4";
    // opt.height = buf.cameraResult.heightPixel;
    // opt.width = buf.cameraResult.widthPixel;
    // opt.frameRate = buf.cameraResult.frameRate;
    // outputStream = std::unique_ptr<videoOutput>(videoOutput::Create(opt));
    // isRecord = true;
    // frameCount =
    //     params.videDuration * buf.cameraResult.frameRate; // 总共需要保存的帧数
    // drawTimes = floor(frameCount / 3);
  }

  inline void recordVideo(int key, int width, int height) {
    // FrameBuf frameBufMessage = backendPtr->pool->read(key);
    // uchar3 *frame;
    // if (drawTimes-- > 0) {
    //   auto image =
    //       std::any_cast<std::shared_ptr<cv::Mat>>(frameBufMessage.read("Mat"));
    //   drawBox(*image, alarmBox, cv::Scalar{255, 0, 0});
    //   frame = reinterpret_cast<uchar3 *>(image->data);
    // } else {
    //   frame = std::any_cast<uchar3 *>(frameBufMessage.read("uchar3*"));
    // }

    // outputStream->Render(frame, width, height);
    // char str[256];
    // sprintf(str, "Video Viewer (%ux%u)", width, height);
    // // update status bar
    // outputStream->SetStatus(str);
    // if (!outputStream->IsStreaming() || --frameCount <= 0) {
    //   isRecord = false;
    //   frameCount = 0;
    //   drawTimes = 0;
    //   outputStream->Close();
    // }
  }

  inline void generateAlarm(queueMessage &buf, std::string const &detail,
                            retBox const &bbox) {
    // 生成本次报警的唯一ID
    buf.alarmResult.alarmVideoDuration = params.videDuration;
    buf.alarmResult.alarmId = generate_hex(16);
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
      if (buf.frameType == "RGB888") {
        cv::cvtColor(showImage, showImage, cv::COLOR_RGB2BGR);
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
    std::experimental::filesystem::create_directories(
        buf.alarmResult.alarmFile);
    std::string imagePath =
        buf.alarmResult.alarmFile + "/" + buf.alarmResult.alarmId + ".jpg";
    // 输出alarm image
    cv::imwrite(imagePath, showImage);
  }
};
} // namespace module
#endif // __METAENGINE_LOGIC_MODULE_H_
