#include "serialModule.h"
// #include "frameMessage.pb.h"

namespace module {
serialModule::serialModule(backend_ptr ptr, std::string const &name,
                           MessageType const &type)
    : Module(ptr, name, type) {}

void serialModule::forward(std::vector<forwardMessage> &message) {
  // tutorial::FrameMessage buf;
  //    char ch = getchar();
  //    tutorial::FrameMessage buf;
  //    bool flag = false;
  //    if (ch == 's' || ch == 'S')
  //    {
  //        buf.set_control("stop");
  //        flag = true;
  //    } else if (ch == 'i' || ch == 'I')
  //    {
  //        buf.set_control("init");
  //        flag = true;
  //    }
  //    autoSend(buf.SerializeAsString());
}
FlowEngineModuleRegister(serialModule, backend_ptr, std::string const &,
                         MessageType const &);
} // namespace module