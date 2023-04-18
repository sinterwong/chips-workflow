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

#include "alarmUtils.h"
#include "module.hpp"
#include <codecvt>
#include <locale>
#include <opencv2/imgproc.hpp>

using common::ModuleConfig;
using common::OCRConfig;

using common::CharsRet;

namespace module {
class LicensePlateModule : Module {

  std::unique_ptr<OCRConfig> config;

public:
  LicensePlateModule(backend_ptr ptr, std::string const &name,
                     MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config = std::make_unique<OCRConfig>(*config_.getParams<OCRConfig>());
  }

  ~LicensePlateModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

private:
  AlarmUtils alarmUtils;
  std::wstring_view charsetsMapping =
      L"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳"
      L"挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ";

  inline void splitMerge(cv::Mat const &image, cv::Mat &output) {
    int h = image.rows;
    int w = image.cols;
    cv::Rect upperRect{0, 0, w, static_cast<int>(5. / 12. * h)};
    cv::Rect lowerRect{0, static_cast<int>(1. / 3. * h), w,
                       h - static_cast<int>(1. / 3. * h)};
    cv::Mat imageUpper;
    cv::Mat imageLower = image(lowerRect);
    cv::resize(image(upperRect), imageUpper,
               cv::Size(imageLower.cols, imageLower.rows));
    cv::hconcat(imageUpper, imageLower, output);
  };

  inline std::string getChars(CharsRet const &charsRet,
                              std::wstring_view const &charsets) const {
    std::wstring chars = L"";
    for (auto it = charsRet.begin(); it != charsRet.end(); ++it) {
      auto c = charsets.at(*it);
      chars += c;
    }
    // 定义一个UTF-8编码的转换器
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    // 将宽字符字符串转换为UTF-8编码的多字节字符串
    std::string str = converter.to_bytes(chars);
    return str;
  }
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
