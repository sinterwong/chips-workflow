/**
 * @file frameDifferenceModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-06-15
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_FRAME_DIFFERENCE_MODULE_H
#define __METAENGINE_FRAME_DIFFERENCE_MODULE_H

#include <array>
#include <opencv2/opencv.hpp>
#include <string>
#include <utility>

#include "backend.h"
#include "frame_difference.h"
#include "module.hpp"

namespace module {
class FrameDifferenceModule : public Module {
private:
  int count = 0;
  solution::FrameDifference fd;

public:
  FrameDifferenceModule(Backend *ptr, const std::string &initName,
                        const std::string &initType);

  ~FrameDifferenceModule();

  virtual void forward(std::vector<forwardMessage> &message) override;
};
} // namespace module
#endif // __METAENGINE_FRAME_DIFFERENCE_MODULE_H
