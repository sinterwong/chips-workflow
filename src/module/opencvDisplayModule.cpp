#include "opencvDisplayModule.h"

namespace module {
OpencvDisplayModule::OpencvDisplayModule(backend_ptr ptr,
                                         std::string const &name,
                                         std::string const &type)
    : Module(ptr, name, type) {}

void OpencvDisplayModule::forward(std::vector<forwardMessage> &message) {
  for (auto &[send, type, buf] : message) {
    assert(type == "stream");
    // int height, width;
    // height = buf.cameraResult.heightPixel;
    // width = buf.cameraResult.widthPixel;

    auto frameBufMessage = ptr->pool->read(buf.key);
    auto framePtr = std::any_cast<cv::Mat>(frameBufMessage.read("Mat"));

    cv::imshow("image", framePtr);
    cv::waitKey(20);
  }
}
FlowEngineModuleRegister(OpencvDisplayModule, backend_ptr, std::string const &,
                         std::string const &);
} // namespace module
