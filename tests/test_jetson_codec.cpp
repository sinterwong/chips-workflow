/**
 * @file jetson_hwtest.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-06-15
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <thread>

#include "boostMessage.h"
#include "jetson/jetsonSourceModule.h"
#include "jetson/jetsonOutputModule.h"

using namespace module;

int main() {
  BoostMessage bus;
  Backend backend(&bus);
  // rtsp://user:passward@114.242.23.39:9201/test
  // /home/wangxt/workspace/projects/flowengine/sample_1080p_h264.mp4
  std::shared_ptr<JetsonSourceModule> cap(new JetsonSourceModule(
      &backend, "/home/wangxt/workspace/projects/flowengine/sample_1080p_h264.mp4",
      1920, 1080, "h264", "Camera",
      "FrameMessage", {}, {"WriteVideo"}));

  std::shared_ptr<JetsonOutputModule> output(
      new JetsonOutputModule(&backend, "out.mp4", "WriteVideo", "FrameMessage", {"Camera"}));

  std::thread th1(&JetsonOutputModule::go, cap);
  std::thread th2(&JetsonOutputModule::go, output);
  th1.join();
  th2.join();

  return 0;
}
