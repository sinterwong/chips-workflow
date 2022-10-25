//
// Created by Wallel on 2022/3/10.
//

#ifndef METAENGINE_SERIALMODULE_H
#define METAENGINE_SERIALMODULE_H

#include <cstring>

#include "backend.h"
// #include "frameMessage.pb.h"
#include "module.hpp"
namespace module {
class serialModule : public Module {
public:
  serialModule(Backend *ptr, const std::string &initName,
               const std::string &initType,
               const std::vector<std::string> &recv = {},
               const std::vector<std::string> &send = {});
  virtual void forward(std::vector<forwardMessage> message) override;
};

} // namespace module
#endif // METAENGINE_SERIALMODULE_H
