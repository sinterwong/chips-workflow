/**
 * @file ocrModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2023-03-30
 *
 * @copyright Copyright (c) 2023
 *
 */
#ifndef __METAENGINE_LICENSE_PLATE_MODULE_H_
#define __METAENGINE_LICENSE_PLATE_MODULE_H_

#include "module.hpp"
#include <opencv2/imgproc.hpp>

using common::CharsRecoConfig;
using common::ModuleConfig;

using common::CharsRet;

namespace module {
class LicensePlateModule : Module {

  std::unique_ptr<CharsRecoConfig> config;

public:
  LicensePlateModule(backend_ptr ptr, std::string const &name,
                     MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config = std::make_unique<CharsRecoConfig>(
        *config_.getParams<CharsRecoConfig>());
  }

  ~LicensePlateModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

private:
  std::string charsMapping;

  inline void splitMerge(cv::Mat const &image, cv::Mat &output) {
    int h = image.rows;
    int w = image.cols;
    cv::Rect upperRect{0, 0, w, (5 / 12 * h)};
    cv::Rect lowerRect{0, 0, w, (1 / 3 * h)};
    cv::Mat imageUpper;
    cv::Mat imageLower = image(lowerRect);
    cv::resize(image(upperRect), imageUpper,
               cv::Size(imageLower.cols, imageLower.rows));
    cv::hconcat(imageUpper, imageLower, output);
  };

  inline std::string getChars(CharsRet const &charsRet) {
    std::string chars = "";
    for (auto &idx : charsRet) {
      chars += charsMapping.at(idx);
    }
    return chars;
  }
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
