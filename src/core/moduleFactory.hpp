/**
 * @file moduleFactory.hpp
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

#ifndef FlowEngineModuleRegister
#define FlowEngineModuleRegister(X, ...)                                       \
  int __type##X = ModuleFactory::registerModule(                         \
      #X, (void *)(&__createModule<X, __VA_ARGS__>));
#endif

template <typename ModuleClass, typename... ArgsType>
std::shared_ptr<ModuleClass> __createModule(ArgsType... args) {
  return std::make_shared<ModuleClass>(args...);
}

class ModuleFactory {
public:
  template <typename BaseClass, typename... ArgType>
  static std::shared_ptr<BaseClass> createModule(std::string const &className,
                                              ArgType... args) {
    typedef std::shared_ptr<BaseClass> (*_CreateFactory)(ArgType...);

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
