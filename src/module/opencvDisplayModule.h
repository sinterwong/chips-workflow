#ifndef METAENGINE_OPENCVDISPLAYMODULE_H
#define METAENGINE_OPENCVDISPLAYMODULE_H

// #include "frameMessage.pb.h"
#include "module.hpp"
#include <opencv2/opencv.hpp>

#define DIRECT true

namespace module {
class OpencvDisplayModule : public Module {
protected:
  cv::Mat frame;

public:
  OpencvDisplayModule(Backend *ptr, const std::string &initName,
                      const std::string &initType);

  virtual void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // METAENGINE_OPENCVDISPLAYMODULE_H
