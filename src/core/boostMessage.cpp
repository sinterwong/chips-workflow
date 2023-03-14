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
#include <shared_mutex>
#include <thread>

MessageBus::returnFlag BoostMessageCheakEmpty(cqueue_ptr const &q) {
  if (q->size_approx() == 0)
    return MessageBus::returnFlag::successWithEmpty;
  else
    return MessageBus::returnFlag::successWithMore;
}

BoostMessage::BoostMessage() {}

bool BoostMessage::registered(std::string const &name) {
  std::lock_guard lk(m);
  name2Queue.insert(std::make_pair(name, std::make_shared<cqueue>(12)));
  FLOWENGINE_LOGGER_INFO("Register {} module", name);
  return true;
}

bool BoostMessage::unregistered(std::string const &name) {
  std::lock_guard lk(m);
  auto iter = name2Queue.find(name);
  if (iter == name2Queue.end())
    FLOWENGINE_LOGGER_ERROR("Unregister {} module is failed!", name);
  return false;
  name2Queue.erase(iter);
  FLOWENGINE_LOGGER_INFO("{} module has unregistered", name);
  return true;
}

bool BoostMessage::send(std::string const &source, std::string const &target,
                        MessageType const &type, queueMessage message) {
  std::shared_lock lk(m);
  auto iter = name2Queue.find(target);
  if (iter == name2Queue.end())
    return false;

  auto q_ptr = iter->second;

  message.send = source;
  message.recv = target;
  message.messageType = type;

  q_ptr->enqueue(message);
  return true;
}

bool BoostMessage::recv(std::string const &name, MessageBus::returnFlag &flag,
                        std::string &send, MessageType &type,
                        queueMessage &message, bool waitFlag) {
  std::shared_lock lk(m);
  auto iter = name2Queue.find(name);

  if (iter == name2Queue.end()) {
    flag = MessageBus::returnFlag::null;
    send = "";
    type = MessageType::None;
    message = queueMessage();
    FLOWENGINE_LOGGER_ERROR("{} is not registered!", name);
    return false;
  }
  cqueue_ptr receiver = iter->second;
  flag = BoostMessageCheakEmpty(receiver);
  do {
    if (receiver->size_approx() > 0) {
      receiver->try_dequeue(message);
      send = message.send;
      type = message.messageType;
      return true;
    }
    if (waitFlag)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
  } while (waitFlag);
  send = "";
  type = MessageType::None;
  message = queueMessage();
  return false;
}
