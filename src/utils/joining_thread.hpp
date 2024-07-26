/**
 * @file joining_thread.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-05-15
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef __FLOWENGINE_JOINING_THREAD_H_
#define __FLOWENGINE_JOINING_THREAD_H_

#include <thread>
#include <utility>

namespace utils {
class joining_thread {
public:
  std::thread t;
  joining_thread() noexcept = default; // 保持默认构造函数的平实性
  template <typename Callback, typename... Args>
  explicit joining_thread(Callback &&func, Args &&...args)
      : t(std::forward<Callback>(func), std::forward<Args>(args)...) {}

  explicit joining_thread(std::thread t_) noexcept : t(std::move(t_)) {}

  joining_thread(joining_thread &&other) noexcept : t(std::move(other.t)) {}

  joining_thread &operator=(joining_thread &&other) noexcept {
    // 当使用移动赋值构造时，首先检查当前的线程是否已经汇合，如果没有汇合则现将其汇合（届时隶属于该线程的任何存储空间会被清除，t不再关联到已结束的线程）
    if (joinable()) {
      join();
    }
    // 汇合之后即可重新赋值
    t = std::move(other.t);
    return *this;
  }

  joining_thread &operator=(std::thread t_) noexcept {
    if (joinable()) {
      join();
    }
    t = std::move(t_);
    return *this;
  }

  ~joining_thread() noexcept {
    if (joinable()) {
      join();
    }
  }

  void swap(joining_thread &other) noexcept { t.swap(other.t); }

  std::thread::id get_id() const noexcept { return t.get_id(); }

  bool joinable() const noexcept { return t.joinable(); }

  void join() { t.join(); }

  void detach() { t.detach(); }

  std::thread &as_thread() noexcept { return t; }

  const std::thread &as_thread() const noexcept { return t; }
};
} // namespace utils
#endif
