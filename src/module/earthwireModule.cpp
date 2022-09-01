/**
 * @file earthwireModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-30
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "earthwireModule.h"
#include <cstdlib>
#include <experimental/filesystem>
#include <sys/stat.h>
#include <unistd.h>
namespace module {

EarthwireModule::EarthwireModule(Backend *ptr,
                                       const std::string &initName,
                                       const std::string &initType,
                                       const common::LogicConfig &logicConfig,
                                       const std::vector<std::string> &recv,
                                       const std::vector<std::string> &send)
    : LogicModule(ptr, initName, initType, logicConfig, recv, send) {}

/**
 * @brief
 * 1. recv 类型：logic, algorithm
 * 2. send 类型：algorithm, logic
 *
 * @param message
 */
void EarthwireModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  if (recvModule.empty()) {
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      // FLOWENGINE_LOGGER_INFO("{} EarthwireModule module was done!", name);
      std::cout << name << "{} EarthwireModule module was done!"
                << std::endl;
      stopFlag.store(true);
      return;
    }

    if (isRecord) {
      if (type != "stream")
        continue;
      // 正处于保存视频状态
      FrameBuf frameBufMessage = backendPtr->pool->read(buf.key);
      auto frame = std::any_cast<uchar3 *>(frameBufMessage.read("uchar3*"));
      outputStream->Render(frame, buf.cameraResult.widthPixel,
                           buf.cameraResult.heightPixel);
      char str[256];
      sprintf(str, "Video Viewer (%ux%u)", buf.cameraResult.widthPixel,
              buf.cameraResult.heightPixel);
      // update status bar
      outputStream->SetStatus(str);
      if (!outputStream->IsStreaming() || --frameCount <= 0) {
        isRecord = false;
        frameCount = 0;
        outputStream->Close();
      }
      continue;
    }

    if (type == "algorithm") {
      // 此处根据 buf.algorithmResult 信息判断是否存在灭火器
      // 如果符合条件就发送至AlarmOutputModule
      for (int i = 0; i < buf.algorithmResult.bboxes.size(); i++) {
        auto &bbox = buf.algorithmResult.bboxes.at(i);
        if (bbox.first != send) {
          continue;
        }

        // std::cout << "classid: " << bbox.second.at(5) << ", "
        //           << "confidence: " << bbox.second.at(4) << std::endl;
        if (bbox.second.at(5) == 0 && bbox.second.at(4) > 0.9) { // 存在报警
          // 生成本次报警的唯一ID
          buf.alarmResult.alarmVideoDuration = params.videDuration;
          buf.alarmResult.alarmId = generate_hex(16);
          buf.alarmResult.alarmFile =
              params.outputDir + "/" + buf.alarmResult.alarmId;
          buf.alarmResult.alarmDetails = "未检测到接地线";
          buf.alarmResult.alarmType = name;
          buf.alarmResult.page = params.page;
          buf.alarmResult.eventId = params.eventId;

          // TODO
          cv::Mat showImage;
          FrameBuf frameBufMessage = backendPtr->pool->read(buf.key);
          auto frame = std::any_cast<std::shared_ptr<cv::Mat>>(
              frameBufMessage.read("Mat"));
          if (frame->empty()) {
            break;
          }
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
          std::pair<std::string, std::array<float, 6>> b{bbox};
          b.first = name;
          // 单独画出报警框
          drawBox(showImage, b);
          // 画出所有结果
          // drawResult(showImage, buf.algorithmResult);
          buf.algorithmResult.bboxes.emplace_back(b);
          std::experimental::filesystem::create_directories(
              buf.alarmResult.alarmFile);
          std::string imagePath = buf.alarmResult.alarmFile + "/" +
                                  buf.alarmResult.alarmId + ".jpg";
          // 输出alarm image
          cv::imwrite(imagePath, showImage);
          sendWithTypes(buf, {"output"});
          if (params.videDuration > 0) {
            // 需要保存视频，在这里初始化
            videoOptions opt;
            opt.resource = buf.alarmResult.alarmFile + "/" +
                           buf.alarmResult.alarmId + ".mp4";
            opt.height = buf.cameraResult.heightPixel;
            opt.width = buf.cameraResult.widthPixel;
            opt.frameRate = buf.cameraResult.frameRate;
            outputStream =
                std::unique_ptr<videoOutput>(videoOutput::Create(opt));
            isRecord = true;
            frameCount = params.videDuration *
                         buf.cameraResult.frameRate; // 总共需要保存的帧数
          }
          break;
        }
      }
    } else if (type == "stream") {
      // 配置算法推理时需要用到的信息
      buf.logicInfo.region = params.region;
      buf.logicInfo.attentionClasses = params.attentionClasses;
      // 不能发送给output
      sendWithTypes(buf, {"algorithm"});
    }
  }
}

FlowEngineModuleRegister(EarthwireModule, Backend *, std::string const &,
                         std::string const &, common::LogicConfig const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &);
} // namespace module
