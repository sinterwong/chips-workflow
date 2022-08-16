/**
 * @file jetsonOutputModule.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-02
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "jetsonOutputModule.h"
namespace module {
JetsonOutputModule::JetsonOutputModule(Backend *ptr, const std::string &uri,
                                       const std::string &initName,
                                       const std::string &initType,
                                       const std::vector<std::string> &recv,
                                       const std::vector<std::string> &send,
                                       const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool) {
  opt.resource = uri;
  outputStream = std::unique_ptr<videoOutput>(videoOutput::Create(opt));
  // outputStream = videoOutput::Create(opt);
  if (!outputStream) {
    LogError("jetson output:  failed to create input stream\n");
    exit(-1);
  }
}

void JetsonOutputModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  for (auto &[send, type, buf] : message) {
    assert(type == "stream");
    FrameBuf frameBufMessage = backendPtr->pool->read(buf.key);
    auto frame = std::any_cast<uchar3 *>(frameBufMessage.read("uchar3*"));
    // std::cout << (frame == nullptr) << std::endl;
    outputStream->Render(frame, buf.width, buf.height);
    char str[256];
    sprintf(str, "Video Viewer (%ux%u)", buf.width, buf.height);
    // update status bar
    outputStream->SetStatus(str);
    // count ++;
    // if (!outputStream->IsStreaming() || count > 200) {
    if (!outputStream->IsStreaming()) {
      // check if the user quit
      LogInfo("jetson output:  is not streaming!\n");
      LogInfo("jetson output:  output steram was done.\n");
      stopFlag.store(true);
    }
  }
}
FlowEngineModuleRegister(JetsonOutputModule, Backend *, std::string const &,
                         std::string const &, std::string const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &,
                         std::vector<std::string> const &);
} // namespace module