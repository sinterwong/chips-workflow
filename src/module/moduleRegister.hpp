/**
 * @file visionRegister.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-07-26
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef __FLOWENGINE_MODULE_NODE_REGISTER_HPP_
#define __FLOWENGINE_MODULE_NODE_REGISTER_HPP_

namespace module {

class ModuleNodeRegistrar {
public:
  static ModuleNodeRegistrar &getInstance() {
    static ModuleNodeRegistrar instance;
    return instance;
  }

  ModuleNodeRegistrar(const ModuleNodeRegistrar &) = delete;
  ModuleNodeRegistrar &operator=(const ModuleNodeRegistrar &) = delete;

private:
  ModuleNodeRegistrar();
};

} // namespace module

#endif