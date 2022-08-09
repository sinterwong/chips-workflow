//
// Created by Wallel on 2022/3/1.
//

#include "cskTrackModule.h"
namespace module {
cskTrackModule::cskTrackModule(Backend *ptr, const std::string &initName,
                               const std::string &initType,
                               const std::vector<std::string> &recv,
                               const std::vector<std::string> &send,
                               const std::vector<std::string> &pool)
    : Module(ptr, initName, initType, recv, send, pool) {}

cskTrackModule::~cskTrackModule() {}

void cskTrackModule::forward(
    std::vector<std::tuple<std::string, std::string, queueMessage>> message) {
  for (auto &[send, type, buf] : message) {
    if (type == "ControlMessage") {
      std::cout << name << "{} CSKTrackModule module was done!" << std::endl;
      return;
    } else if (type == "FrameMessage") {
      if (moduleFlag == init || moduleFlag == tracking) {
        auto frameBufMessage = backendPtr->pool->read(buf.key);

        cv::Mat image = std::any_cast<cv::Mat>(frameBufMessage.read("Mat"));

        //                if (moduleFlag == tracking)
        //                {
        //                    std::tie(answer, value) = tracker.update(image);
        //                } else
        //                {
        //                    tracker.init(image, floatArrayToCvRect(roi));
        //                    answer = roi;
        //                    moduleFlag = tracking;
        //                }

        cv::rectangle(image, floatArrayToCvRect(roi), cv::Scalar(0, 0, 255), 2);
        //                std::cout << roi[0] << " "
        //                          << roi[0] + roi[2] << " "
        //                          << roi[1] << " "
        //                          << roi[1] + roi[3] << std::endl;
        //                for (int i = roi[1]; i < roi[1] + roi[3]; i++)
        //                {
        //                    for (int j = roi[0]; j < roi[0] + roi[2]; j++)
        //                    {
        //                        frameBufMessage.data[i * frameBufMessage.width
        //                        * 3 +
        //                                             j * 3 + 0] = 0;
        //                        frameBufMessage.data[i * frameBufMessage.width
        //                        * 3 +
        //                                             j * 3 + 1] = 0;
        //                        frameBufMessage.data[i * frameBufMessage.width
        //                        * 3 +
        //                                             j * 3 + 2] = 255;
        //                    }
        //                }
      }
      autoSend(buf);
    }
  }
}

cv::Rect cskTrackModule::floatArrayToCvRect(const std::array<float, 4> &a) {
  return cv::Rect(a[0], a[1], a[2], a[3]);
}
} // namespace module
