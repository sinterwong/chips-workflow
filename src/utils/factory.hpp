/**
 * @file factory.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief object factory
 * @version 0.1
 * @date 2022-08-14
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __DESIGN_PATTERN_REFLECTION_H_
#define __DESIGN_PATTERN_REFLECTION_H_
#include <map>
#include <memory>
#include <string>

namespace utils {
#ifndef FlowEngineModuleRegister
#define FlowEngineModuleRegister(X, ...)                                       \
  __attribute__((used)) static int __type##X =                                 \
      ObjectFactory::regCreateObjFunc(                                         \
          #X,                                                                  \
          (void *)(&CreateObjHelper<X __VA_OPT__(, ) __VA_ARGS__>::create));
#endif

template <typename YourClass, typename... ArgType> struct CreateObjHelper {
  static std::shared_ptr<YourClass> create(ArgType... arg) {
    return std::make_shared<YourClass>(std::forward<ArgType>(arg)...);
  }
};

template <typename YourClass> std::shared_ptr<YourClass> __createObjectFunc() {
  return CreateObjHelper<YourClass>::create();
}

template <typename YourClass, typename... ArgType>
std::shared_ptr<YourClass> __createObjectFunc(ArgType &&...arg) {
  return CreateObjHelper<YourClass, ArgType...>::create(
      std::forward<ArgType>(arg)...);
}

class ObjectFactory {
public:
  template <class BaseClass, typename... ArgType>
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

  static int regCreateObjFunc(std::string const &className, void *func) {
    _GetStaticFuncMap()[className] = func;
    return 0;
  }

private:
  static std::map<std::string const, void *> &_GetStaticFuncMap() {
    static std::map<std::string const, void *> _classMap;
    return _classMap;
  }
};
} // namespace utils
#endif
