/**
 * @file extinguisherModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "extinguisherModule.h"
#include <cstdlib>
#include <experimental/filesystem>
#include <sys/stat.h>
#include <unistd.h>

namespace module {

ExtinguisherModule::ExtinguisherModule(Backend *ptr, const std::string &initName,
                             const std::string &initType,
                             const common::LogicConfig &logicConfig,
                             const std::vector<std::string> &recv,
                             const std::vector<std::string> &send,
                             const std::vector<std::string> &pool)
    : LogicModule(ptr, initName, initType, logicConfig, recv, send, pool) {}

void ExtinguisherModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  if (recvModule.empty()) {
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      // FLOWENGINE_LOGGER_INFO("{} ExtinguisherModule module was done!", name);
      std::cout << name << "{} ExtinguisherModule module was done!" << std::endl;
      buf.status = 1;
      stopFlag.store(true);
      if (outputStream && outputStream->IsStreaming()) {
        outputStream->Close();
      }
      return;
    }
    if (isRecord) {
      // 正处于保存视频状态
      FrameBuf frameBufMessage = backendPtr->pool->read(buf.key);
      auto frame = std::any_cast<uchar3 *>(frameBufMessage.read("uchar3*"));
      outputStream->Render(frame, buf.width, buf.height);
      char str[256];
      sprintf(str, "Video Viewer (%ux%u)", buf.width, buf.height);
      // update status bar
      outputStream->SetStatus(str);
      if (!outputStream->IsStreaming() || --frameCount <= 0) {
        isRecord = false;
        frameCount = 0;
        outputStream->Close();
        return;
      }
      if (type != "stream") {
        // 当前状态只接受FrameMessage消息
        return;
      }

      return;
    }

    if (type == "algorithm") {
      // 此处根据 buf.algorithmResult 信息判断是否存在灭火器
      // 如果符合条件就发送至AlarmOutputModule
      for (int i = 0; i < buf.algorithmResult.bboxes.size(); i++) {
        auto &bbox = buf.algorithmResult.bboxes.at(i);
        if (bbox.first == send) {
          // std::cout << "classid: " << bbox.second.at(5) << ", "
          //           << "confidence: " << bbox.second.at(4) << std::endl;
          if (bbox.second.at(5) != 5 && bbox.second.at(4) > 0.93) { // 存在报警
            // 生成本次报警的唯一ID
            buf.alarmResult.alarmVideoDuration = params.videDuration;
            buf.alarmResult.alarmId = generate_hex(16);
            buf.alarmResult.alarmFile =
                params.outputDir + "/" + buf.alarmResult.alarmId;
            buf.alarmResult.alarmDetails = "不存在灭火器";
            buf.alarmResult.alarmType = name;
            buf.alarmResult.page = params.page;
            buf.alarmResult.eventId = params.eventId;

            // TODO
            // 临时画个图（后续根据前端参数来决定返回的图片是否带有画图标记）
            FrameBuf frameBufMessage = backendPtr->pool->read(buf.key);
            auto frame = std::any_cast<std::shared_ptr<cv::Mat>>(
                frameBufMessage.read("Mat"));
            if (frame->empty()) {
              return;
            }
            cv::Mat showImage = frame->clone();
            if (buf.frameType == "RGB888") {
              cv::cvtColor(showImage, showImage, cv::COLOR_RGB2BGR);
            }

            // 记录当前的框为报警框
            std::pair<std::string, std::array<float, 6>> b{bbox};
            b.first = name;
            // 单独画出报警框
            drawBox(showImage, b);
            buf.algorithmResult.bboxes.emplace_back(b);

            std::experimental::filesystem::create_directories(
                buf.alarmResult.alarmFile);
            std::string imagePath = buf.alarmResult.alarmFile + "/" +
                                    buf.alarmResult.alarmId + ".jpg";
            cv::imwrite(imagePath, showImage);
            autoSend(buf);

            if (params.videDuration > 0) {
              // 需要保存视频，在这里初始化
              videoOptions opt;
              opt.resource = buf.alarmResult.alarmFile + "/" +
                             buf.alarmResult.alarmId + ".mp4";
              opt.height = buf.height;
              opt.width = buf.width;
              opt.frameRate = 25;
              outputStream =
                  std::unique_ptr<videoOutput>(videoOutput::Create(opt));
              isRecord = true;
              frameCount = params.videDuration * 25; // 总共需要保存的帧数
            }
            break;
          }
        }
      }
    } else if (type == "stream") {
      // 配置算法推理时需要用到的信息
      buf.logicInfo.region = params.region;
      buf.logicInfo.attentionClasses = params.attentionClasses;
      // 不能发送给output
      sendWithoutTypes(buf, {"output"});
    }
  }
}

FlowEngineModuleRegister(ExtinguisherModule, Backend *, std::string const &,
                         std::string const &, common::LogicConfig const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &);
} // namespace module
