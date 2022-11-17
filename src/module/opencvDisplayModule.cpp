#include "opencvDisplayModule.h"

namespace module {
OpencvDisplayModule::OpencvDisplayModule(Backend *ptr,
                                         const std::string &initName,
                                         const std::string &initType)
    : Module(ptr, initName, initType) {}

void OpencvDisplayModule::forward(std::vector<forwardMessage> message) {
  for (auto &[send, type, buf] : message) {
    assert(type == "stream");
    // int height, width;
    // height = buf.cameraResult.heightPixel;
    // width = buf.cameraResult.widthPixel;

    auto frameBufMessage = backendPtr->pool->read(buf.key);
    auto framePtr = std::any_cast<cv::Mat>(frameBufMessage.read("Mat"));

    cv::imshow("image", framePtr);
    cv::waitKey(20);
  }
}
FlowEngineModuleRegister(OpencvDisplayModule, Backend *, std::string const &,
                         std::string const &);
} // namespace module
