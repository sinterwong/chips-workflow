/**
 * @file jetsonOutputModule.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-06-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef __METAENGINE_JETSON_OUTPUT_H
#define __METAENGINE_JETSON_OUTPUT_H

#include <any>
#include <memory>
#include <vector>

#include "module.hpp"
#include "videoOutput.h"

namespace module {
class JetsonOutputModule : public Module {
private:
  bool ret;

  int count = 0;

  videoOptions opt;

  // std::unique_ptr<videoOutput> outputStream;
  videoOutput *outputStream;

public:
  JetsonOutputModule(Backend *ptr, const std::string &uri,
                     const std::string &initName, const std::string &initType,
                     const std::vector<std::string> &recv = {},
                     const std::vector<std::string> &send = {},
                     const std::vector<std::string> &pool = {});
  ~JetsonOutputModule() {
    delete outputStream;
    outputStream = nullptr;
  }

  void forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
                   message) override;
};
} // namespace module
#endif // __METAENGINE_JETSON_OUTPUT_H
