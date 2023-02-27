#ifndef METAENGINE_SERIALMODULE_H
#define METAENGINE_SERIALMODULE_H

#include <cstring>

#include "backend.h"
// #include "frameMessage.pb.h"
#include "module.hpp"
namespace module {
class serialModule : public Module {
public:
  serialModule(backend_ptr ptr, std::string const &name,
               std::string const &type);
  virtual void forward(std::vector<forwardMessage> &message) override;
};

} // namespace module
#endif // METAENGINE_SERIALMODULE_H
