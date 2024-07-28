/**
 * @file visionRegister.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-26
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "moduleRegister.hpp"
#include "utils/factory.hpp"

#include "alarmOutputModule.hpp"
#include "detClsModule.hpp"
#include "frameDifferenceModule.hpp"
#include "licensePlateModule.hpp"
#include "objectCounterModule.hpp"
#include "objectNumberModule.hpp"
#include "statusOutputModule.hpp"
#include "streamModule.hpp"

using namespace utils;

namespace module {
ModuleNodeRegistrar::ModuleNodeRegistrar() {
  FlowEngineModuleRegister(AlarmOutputModule, backend_ptr, std::string const &,
                           MessageType const &, ModuleConfig &);
  FlowEngineModuleRegister(DetClsModule, backend_ptr, std::string const &,
                           MessageType const &, ModuleConfig &);
  FlowEngineModuleRegister(FrameDifferenceModule, backend_ptr,
                           std::string const &, MessageType &, ModuleConfig &);
  FlowEngineModuleRegister(LicensePlateModule, backend_ptr, std::string const &,
                           MessageType const &, ModuleConfig &);
  FlowEngineModuleRegister(ObjectCounterModule, backend_ptr,
                           std::string const &, MessageType const &,
                           ModuleConfig &);
  FlowEngineModuleRegister(ObjectNumberModule, backend_ptr, std::string const &,
                           MessageType const &, ModuleConfig &);
  FlowEngineModuleRegister(StatusOutputModule, backend_ptr, std::string const &,
                           MessageType const &, ModuleConfig &);
}

} // namespace module