#include "cskTrackModule.h"
namespace module {
cskTrackModule::cskTrackModule(backend_ptr ptr, std::string const &name,
                               std::string const &type)
    : Module(ptr, name, type) {}

cskTrackModule::~cskTrackModule() {}

void cskTrackModule::forward(std::vector<forwardMessage> &message) {
  // for (auto &[send, type, buf] : message) {
  //   if (type == "ControlMessage") {
  //     std::cout << name << "{} CSKTrackModule module was done!" << std::endl;
  //     return;
  //   } else if (type == "stream") {
  //     if (moduleFlag == init || moduleFlag == tracking) {
  //       auto frameBufMessage = ptr->pool->read(buf.key);

  //       cv::Mat image = std::any_cast<cv::Mat>(frameBufMessage.read("Mat"));

  //       //                if (moduleFlag == tracking)
  //       //                {
  //       //                    std::tie(answer, value) =
  //       tracker.update(image);
  //       //                } else
  //       //                {
  //       //                    tracker.init(image, floatArrayToCvRect(roi));
  //       //                    answer = roi;
  //       //                    moduleFlag = tracking;
  //       //                }

  //       cv::rectangle(image, floatArrayToCvRect(roi), cv::Scalar(0, 0, 255),
  //       2);
  //       //                std::cout << roi[0] << " "
  //       //                          << roi[0] + roi[2] << " "
  //       //                          << roi[1] << " "
  //       //                          << roi[1] + roi[3] << std::endl;
  //       //                for (int i = roi[1]; i < roi[1] + roi[3]; i++)
  //       //                {
  //       //                    for (int j = roi[0]; j < roi[0] + roi[2]; j++)
  //       //                    {
  //       //                        frameBufMessage.data[i *
  //       frameBufMessage.width
  //       //                        * 3 +
  //       //                                             j * 3 + 0] = 0;
  //       //                        frameBufMessage.data[i *
  //       frameBufMessage.width
  //       //                        * 3 +
  //       //                                             j * 3 + 1] = 0;
  //       //                        frameBufMessage.data[i *
  //       frameBufMessage.width
  //       //                        * 3 +
  //       //                                             j * 3 + 2] = 255;
  //       //                    }
  //       //                }
  //     }
  //     autoSend(buf);
  //   }
  // }
}

cv::Rect cskTrackModule::floatArrayToCvRect(const std::array<float, 4> &a) {
  return cv::Rect(a[0], a[1], a[2], a[3]);
}
FlowEngineModuleRegister(cskTrackModule, backend_ptr, std::string const &,
                         std::string const &);
} // namespace module
