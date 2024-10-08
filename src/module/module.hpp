/**
 * @file module.hpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.3
 * @date 2022-09-05
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef FLOWCORE_MODULE_HPP
#define FLOWCORE_MODULE_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "backend.hpp"
#include "logger/logger.hpp"
#include "utils/factory.hpp"

using common::MessageType;
using common::ModuleBase;
using namespace utils;

namespace module {
using forwardMessage = std::tuple<std::string, MessageType, queueMessage>;
using backend_ptr = std::shared_ptr<Backend>;

class Module {
protected:
  backend_ptr ptr;

  std::string name; // 模块名称

  MessageType type; // 模块类型 {stream, output, logic}

  ModuleBase config; // 模块类型 {stream, output, logic}

  std::vector<std::string> recvModule, sendModule;

public:
  std::atomic_bool stopFlag;

  Module(backend_ptr ptr_, std::string const &name_, MessageType const &type_,
         ModuleBase config)
      : ptr(ptr_), name(name_), type(type_), config(config) {
    stopFlag.store(false);
    ptr->message->registered(name);
  }
  virtual ~Module() { ptr->message->unregistered(name); }

  virtual bool isRunning() { return !stopFlag.load(); };

  virtual void beforeGetMessage(){};

  virtual void beforeForward(){};

  virtual void forward(std::vector<forwardMessage> &message) = 0;

  virtual void afterForward(){};

  virtual void go() {
    while (!stopFlag.load()) {
      step();
      std::this_thread::sleep_for(std::chrono::milliseconds(config.interval));
    }
  }

  virtual void step() {
    beforeGetMessage();

    // 收集目前已经收到的消息，因此需要多次访问recv
    std::unordered_map<std::string, forwardMessage> mselector;
    bool loop = false;
    do {
      std::string sender;
      MessageType stype;
      queueMessage message;
      loop = ptr->message->recv(name, sender, stype, message);
      if (loop) { // 成功接收到了消息
        // 如果存在同一发送者的消息，只处理一次，用新的去覆盖旧的即可
        mselector[sender] = std::make_tuple(std::move(sender), std::move(stype),
                                            std::move(message));
      }
    } while (loop);

    if (mselector.empty()) {
      if (recvModule.size() == 1) {
        // 只有Administrator的情况下就让它睡一秒来空出cpu
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      return;
    }

    // 获取到了执行任务，开始forward
    beforeForward();
    // 将收集到的消息转换成待处理消息后去处理
    std::vector<forwardMessage> messages;
    messages.reserve(mselector.size());
    std::transform(std::make_move_iterator(mselector.begin()),
                   std::make_move_iterator(mselector.end()),
                   std::back_inserter(messages),
                   [](auto &&pair) { return std::move(pair.second); });
    forward(messages);
    afterForward();
  }

  /**
   * @brief 向指定的类型发送消息
   *
   * @param buf
   * @param types_
   * @return true
   * @return false
   */
  bool sendWithTypes(queueMessage const &buf,
                     std::vector<std::string> const &types_) {
    bool ret = false;
    for (auto &target : sendModule) {
      std::string sendtype_ = target.substr(0, target.find("_"));
      auto iter = std::find(types_.begin(), types_.end(), sendtype_);
      if (iter != types_.end()) {
        // 意味着在不发送的列表中找到了该类型
        auto temp = ptr->message->send(name, target, type, buf);
        ret = ret and temp;
      }
    }
    return ret;
  }

  bool autoSend(const queueMessage &buf) {
    bool ret = false;
    for (auto &target : sendModule) {
      auto temp = ptr->message->send(name, target, type, buf);
      ret = ret and temp;
    }
    return ret;
  }

  void delSendModule(std::string const &name) {
    auto iter = std::remove(sendModule.begin(), sendModule.end(), name);
    if (iter != sendModule.end()) {
      sendModule.erase(iter, sendModule.end());
    }
  }

  void addSendModule(std::string const &name) {
    auto iter = std::find(sendModule.begin(), sendModule.end(), name);
    if (iter == sendModule.end()) {
      sendModule.push_back(name);
    }
  }

  void delRecvModule(std::string const &name) {
    auto iter = std::remove(recvModule.begin(), recvModule.end(), name);
    if (iter != recvModule.end()) {
      recvModule.erase(iter, recvModule.end());
    }
  }

  void addRecvModule(std::string const &name) {
    auto iter = std::find(recvModule.begin(), recvModule.end(), name);
    if (iter == recvModule.end()) {
      recvModule.push_back(name);
    }
  }

  std::vector<std::string> getSendModule() { return sendModule; }

  std::vector<std::string> getRecvModule() { return recvModule; }
};
} // namespace module
#endif // FLOWCORE_MODULE_HPP
