#ifndef __FLOWENGINE_MODULE_PARAMETER_CENTER_HPP_
#define __FLOWENGINE_MODULE_PARAMETER_CENTER_HPP_

#include "common/module_header.hpp"

using namespace common;
namespace module {
// 定义一个参数中心
class ModuleConfig {
public:
  // 将所有参数类型存储在一个 std::variant 中
  using Params = std::variant<StreamBase, OutputBase, WithoutHelmet,
                              SmokingMonitor, ExtinguisherMonitor>;

  // 设置参数
  template <typename T> void setParams(T params) {
    params_ = std::move(params);
  }

  // 访问参数
  template <typename Func> void visitParams(Func &&func) {
    std::visit([&](auto &&params) { std::forward<Func>(func)(params); },
               params_);
  }

  // 获取参数
  template <typename T> T *getParams() { return std::get_if<T>(&params_); }

private:
  Params params_;
};

} // namespace module

#endif