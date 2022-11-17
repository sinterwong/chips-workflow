/**
 * @file oiltubeModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-30
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __METAENGINE_OILTUBE_MODULE_H_
#define __METAENGINE_OILTUBE_MODULE_H_

#include <any>
#include <memory>
#include <vector>

#include "common/common.hpp"
#include "logger/logger.hpp"
#include "logicModule.h"

namespace module {
class OiltubeModule : public LogicModule {

public:
  OiltubeModule(Backend *ptr, const std::string &initName,
                const std::string &initType,
                const common::LogicConfig &logicConfig);
  ~OiltubeModule() {}

  virtual void forward(std::vector<forwardMessage> message) override;
};
} // namespace module
#endif // __METAENGINE_Oiltube_MODULE_H_
