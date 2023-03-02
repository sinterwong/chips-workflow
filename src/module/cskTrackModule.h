#ifndef METAENGINE_CSKTRACKMODULE_H
#define METAENGINE_CSKTRACKMODULE_H

#include <array>
#include <opencv2/opencv.hpp>
#include <string>

#include "backend.h"
// #include "frameMessage.pb.h"
#include "csk.hpp"
#include "module.hpp"

namespace module {
class cskTrackModule : public Module {
private:
  enum cskFlag {
    stop,
    init,
    tracking,
  };

  cskFlag moduleFlag = init;
  CSK::CSKTracker tracker;
  std::array<float, 4> roi = {1920.0 / 2 - 200.0 / 2, 1080.0 / 2 - 200.0 / 2,
                              200, 200};
  std::array<float, 4> answer;
  float value;

  cv::Rect floatArrayToCvRect(const std::array<float, 4> &a);

public:
  cskTrackModule(backend_ptr ptr, std::string const &name,
                 MessageType const &type);

  ~cskTrackModule();

  void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // METAENGINE_CSKTRACKMODULE_H
