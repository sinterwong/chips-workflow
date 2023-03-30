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

#ifndef __METAENGINE_OCR_GENERAL_MODULE_H_
#define __METAENGINE_OCR_GENERAL_MODULE_H_

#include "module.hpp"

using common::CharsRecoConfig;
using common::ModuleConfig;

namespace module {
class CharsRecognitionModule : Module {

  std::unique_ptr<CharsRecoConfig> config;

public:
  CharsRecognitionModule(backend_ptr ptr, std::string const &name,
                         MessageType const &type, ModuleConfig &config_)
      : Module(ptr, name, type) {
    config = std::make_unique<CharsRecoConfig>(
        *config_.getParams<CharsRecoConfig>());
  }

  ~CharsRecognitionModule() {}

  virtual void forward(std::vector<forwardMessage> &message) override;

private:
  std::string chars;
};
} // namespace module
#endif // __METAENGINE_HELMET_MODULE_H_
