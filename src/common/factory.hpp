/**
 * @file factory.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-16
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef __MODULE_FACTORY_REFLECTION_H_
#define __MODULE_FACTORY_REFLECTION_H_

#include <map>
#include <memory>
#include <iostream>

#ifndef FlowEngineModuleRegister
#define FlowEngineModuleRegister(X, ...)                                       \
  static int __type##X = ObjectFactory::registerModule(                               \
      #X, (void *)(&__createObject<X, __VA_ARGS__>));
#endif

template <typename ModuleClass, typename... ArgsType>
std::shared_ptr<ModuleClass> __createObject(ArgsType... args) {
  return std::make_shared<ModuleClass>(args...);
}

class ObjectFactory {
public:
  template <typename BaseClass, typename... ArgType>
  static std::shared_ptr<BaseClass> createObject(std::string const &className,
                                                 ArgType... args) {
    using _CreateFactory = std::shared_ptr<BaseClass> (*)(ArgType...);

    auto &_funcMap = _GetStaticFuncMap();
    auto iFind = _funcMap.find(className);
    if (iFind == _funcMap.end()) {
      return nullptr;
    } else {
      return reinterpret_cast<_CreateFactory>(_funcMap[className])(args...);
    }
  }

  static int registerModule(std::string const &className, void *func) {
    _GetStaticFuncMap()[className] = func;
    return 0;
  }

private:
  static std::map<std::string const, void *> &_GetStaticFuncMap() {
    static std::map<std::string const, void *> _classMap;
    return _classMap;
  }
};
#endif // __MODULE_FACTORY_REFLECTION_H_
