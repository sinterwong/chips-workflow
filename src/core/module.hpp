//
// Created by Wallel on 2022/2/1.
//

#ifndef FLOWCORE_MODULE_HPP
#define FLOWCORE_MODULE_HPP

#include <atomic>
#include <memory>
#include <vector>

#include "backend.h"

namespace module {
class Module {
protected:
  Backend *backendPtr;

  std::string name;

  std::vector<std::string> framePool;

  std::string type;

  std::unordered_map<std::string, int> hash;

  std::vector<std::tuple<std::string, std::string, queueMessage>> message;

  bool loop;

  std::mutex _m;

public:
  std::vector<std::string> recvModule, sendModule;

  std::atomic_bool stopFlag;

  Module(Backend *ptr, const std::string &initName, const std::string &initType,
         const std::vector<std::string> &recv = {},
         const std::vector<std::string> &send = {},
         const std::vector<std::string> &pool = {}) {
    backendPtr = ptr;
    name = initName;
    type = initType;
    recvModule = recv;
    sendModule = send;
    framePool = pool;

    stopFlag.store(false);
    backendPtr->message->registered(name);
  }

  virtual void beforeGetMessage(){};

  virtual void beforeForward(){};

  virtual void
  forward(std::vector<std::tuple<std::string, std::string, queueMessage>>
              message) = 0;

  virtual void afterForward(){};

  void go() {
    while (!stopFlag.load()) {
      step();
    }
  }

  void step() {
    message.clear();
    hash.clear();
    loop = false;

    beforeGetMessage();
    if (!recvModule.empty()) {
      while (message.empty() || loop) {
        MessageBus::returnFlag flag;
        std::string Msend, Mtype;
        queueMessage Mbyte;
        backendPtr->message->recv(name, flag, Msend, Mtype, Mbyte, true);
#ifdef DEBUG
        assert(flag == MessageBus::returnFlag::successWithMore ||
               flag == MessageBus::returnFlag::successWithEmpty);
#endif
        auto iter = hash.find(Msend);
        if (iter == hash.end()) {
          hash.insert(std::make_pair(Msend, message.size()));
          message.emplace_back(std::make_tuple(Msend, Mtype, Mbyte));
        } else {
          auto key = iter->second;
          message[key] = std::make_tuple(Msend, Mtype, Mbyte);
        }
        loop = flag == MessageBus::returnFlag::successWithMore;
      }
      // std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    beforeForward();

    forward(message);
    afterForward();
  }

  bool autoSend(const queueMessage &buf) {
    bool ret = false;
    for (auto &target : sendModule) {
      auto temp = backendPtr->message->send(name, target, type, buf);
      ret = ret and temp;
    }
    return ret;
  }

  void delSendModule(std::string const &name) {
    std::lock_guard<std::mutex> lk(_m);
    auto iter = std::remove(sendModule.begin(), sendModule.end(), name);
    sendModule.erase(iter, sendModule.end());
  }

  void addSendModule(std::string const &name) {
    std::lock_guard<std::mutex> lk(_m);
    sendModule.push_back(name);
  }

  void delRecvModule(std::string const &name) {
    std::lock_guard<std::mutex> lk(_m);
    auto iter = std::remove(recvModule.begin(), recvModule.end(), name);
    recvModule.erase(iter, recvModule.end());
  }

  void addRecvModule(std::string const &name) {
    std::lock_guard<std::mutex> lk(_m);
    recvModule.push_back(name);
  }
};
} // namespace module
#endif // FLOWCORE_MODULE_HPP
