//
// Created by Wallel on 2022/2/22.
//
#include <thread>

#include "boostMessage.h"
#include "opencvCameraModule.h"
#include "opencvDisplayModule.h"

using namespace module;

int main() {
  BoostMessage bus;
  Backend backend(&bus);

  std::shared_ptr<OpencvCameraModule> cap(new OpencvCameraModule(
      &backend, "/mnt/d/Code/flowengine/resource/fire.mp4", "Camera",
      "FrameMessage", {}, {"Display"}));

  std::shared_ptr<OpencvDisplayModule> display(
      new OpencvDisplayModule(&backend, "Display", "FrameMessage", {"Camera"}));

  std::thread th1(&OpencvCameraModule::go, cap);
  std::thread th2(&OpencvDisplayModule::go, display);

  th1.join();
  th2.join();

  return 0;
}