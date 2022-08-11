/**
 * @file CallingModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-07-31
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "callingModule.h"
#include <experimental/filesystem>
#include <sys/stat.h>
#include <unistd.h>

namespace module {

CallingModule::CallingModule(Backend *ptr, const std::string &initName,
                             const std::string &initType,
                             const common::LogicConfig &logicConfig,
                             const std::vector<std::string> &recv,
                             const std::vector<std::string> &send,
                             const std::vector<std::string> &pool)
    : LogicModule(ptr, initName, initType, logicConfig, recv, send, pool) {}

void CallingModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  if (recvModule.empty()) {
    return;
  }
  for (auto &[send, type, buf] : message) {
    if (isRecord) {
      // 正处于保存视频状态
      FrameBuf frameBufMessage = backendPtr->pool->read(buf.key);
      auto frame = std::any_cast<uchar3 *>(frameBufMessage.read("uchar3*"));
      outputStream->Render(frame, buf.width, buf.height);
      char str[256];
      sprintf(str, "Video Viewer (%ux%u)", buf.width, buf.height);
      // update status bar
      outputStream->SetStatus(str);
      if (!outputStream->IsStreaming() || --frameCount <= 0 ) {
        isRecord = false;
        frameCount = 0;
        outputStream->Close();
        return;
      }
      if (type != "FrameMessage") {
        // 当前状态只接受FrameMessage消息
        return ;
      }
      
      return; 
    }
    if (type == "ControlMessage") {
      // FLOWENGINE_LOGGER_INFO("{} CallingModule module was done!", name);
      std::cout << name << "{} CallingModule module was done!" << std::endl;
      stopFlag.store(true);
      return;
    } else if (type == "AlgorithmMessage") {
      // 此处根据 buf.algorithmResult 写吸烟的逻辑并填充 buf.alarmResult 信息
      // 如果符合条件就发送至SendOutputModule
      for (int i = 0; i < buf.algorithmResult.bboxes.size(); i++) {
        auto &bbox = buf.algorithmResult.bboxes.at(i);
        if (bbox.first == send) {
          std::cout << "classid: " << bbox.second.at(5) << ", "
                    << "confidence: " << bbox.second.at(4) << std::endl;
          if (bbox.second.at(5) != 0) { // 存在报警

            // 生成本次报警的唯一ID
            buf.alarmResult.alarmVideoDuration = params.videDuration;
            buf.alarmResult.alarmId = generate_hex(16);
            buf.alarmResult.alarmFile =
                params.outputDir + "/" + buf.alarmResult.alarmId;
            buf.alarmResult.alarmDetails = "存在吸烟或打电话";
            buf.alarmResult.alarmType = name;

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
            drawResult(showImage, buf.algorithmResult);

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
              outputStream = std::unique_ptr<videoOutput>(videoOutput::Create(opt));
              isRecord = true;
              frameCount = params.videDuration * 25; // 总共需要保存的帧数
            }
            break;
          }
        }
      }
    }
  }
}
} // namespace module
