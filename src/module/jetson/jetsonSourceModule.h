/**
 * @file jetsonSourceModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-06-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __METAENGINE_JETSON_SOURCE_H
#define __METAENGINE_JETSON_SOURCE_H

#include <any>
#include <memory>
#include <vector>

#include "common/common.hpp"
#include "frameMessage.pb.h"
#include "module.hpp"
#include "videoSource.h"
#include <opencv2/opencv.hpp>
namespace module {

void delBuffer(std::vector<std::any> &);
std::any getPtrBuffer(std::vector<std::any> &, FrameBuf *);
std::any getBuffer(std::vector<std::any> &, FrameBuf *);
FrameBuf makeFrameBuf(uchar3 *, int, int);

class JetsonSourceModule : public Module {
private:
  std::string uri;

  // std::unique_ptr<videoSource> inputStream;
  videoSource *inputStream;

  uchar3 *frame = nullptr;

  int count = 0;

  bool ret;

  tutorial::FrameMessage buf;

  bool signal_recieved;

  videoOptions opt;

public:
  JetsonSourceModule(Backend *ptr,
                     const std::string &initName, const std::string &initType,
                     const common::CameraConfig &_params,
                     const std::vector<std::string> &recv = {},
                     const std::vector<std::string> &send = {},
                     const std::vector<std::string> &pool = {});

  ~JetsonSourceModule() {
    delete inputStream;
    inputStream = nullptr;
    // delete [] frame;
    // frame = nullptr;
  }

  void forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
                   message) override;
  
};
} // namespace module
#endif // __METAENGINE_JETSON_SOURCE_H
