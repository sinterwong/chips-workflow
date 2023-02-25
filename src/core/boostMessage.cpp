/**
 * @file boostMessage.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-07-13
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "boostMessage.h"
#include "logger/logger.hpp"
#include "messageBus.h"
#include <memory>
#include <mutex>
#include <thread>

MessageBus::returnFlag BoostMessageCheakEmpty(
    const std::shared_ptr<moodycamel::ConcurrentQueue<queueMessage>> &q) {
  if (q->size_approx() == 0)
    return MessageBus::returnFlag::successWithEmpty;
  else
    return MessageBus::returnFlag::successWithMore;
}

BoostMessage::BoostMessage() {}

bool BoostMessage::registered(std::string name) {
  FLOWENGINE_LOGGER_INFO("Create module: {}", name);
  socketRecv.emplace_back(
      std::make_shared<moodycamel::ConcurrentQueue<queueMessage>>(16));

  int location = socketRecv.size() - 1;
  name2Queue.insert(std::make_pair(name, location));

  return true;
}

bool BoostMessage::send(std::string source, std::string target,
                        std::string type, queueMessage message) {
  auto iter = name2Queue.find(target);
  if (iter == name2Queue.end())
    return false;

  int key = iter->second;

  message.send = source;
  message.recv = target;
  message.messageType = type;

  socketRecv[key]->enqueue(message);
  return true;
}

bool BoostMessage::recv(std::string source, MessageBus::returnFlag &flag,
                        std::string &send, std::string &type,
                        queueMessage &byte, bool waitFlag) {
  auto iter = name2Queue.find(source);

  if (iter == name2Queue.end()) {
    flag = MessageBus::returnFlag::null;
    send = "";
    type = "";
    byte = queueMessage();
    return false;
  }

  // auto &recver {socketRecv[iter->second]};
  auto recver{socketRecv[iter->second]};

  flag = BoostMessageCheakEmpty(recver);

  // std::cout << source << "'s thread id: " << std::this_thread::get_id() <<
  // std::endl; std::cout << source << "'s recver id: " << recver.get() <<
  // std::endl;
  if (recver == nullptr) {
    FLOWENGINE_LOGGER_WARN("{} recver is nullptr!", source);
    return false;
  }
  assert(recver);

  bool first = true;

  while (first || waitFlag) {
    // if (recver->size_approx() > 0) {
    if (recver->size_approx() > 0) {
      recver->try_dequeue(byte);
      send = byte.send;
      type = byte.messageType;
      return true;
    }
    if (waitFlag)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    first = false;
  }
  flag = MessageBus::returnFlag::null;
  send = "";
  type = "";
  byte = queueMessage();
  return false;
}
