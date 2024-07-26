/**
 * @file boostMessage.h
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.2
 * @date 2022-08-05
 *
 * @copyright Copyright (c) 2022
 *
 */
#ifndef FLOWCORE_BOOSTMESSAGE_H
#define FLOWCORE_BOOSTMESSAGE_H

#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

#include "utils/concurrentqueue.h"
#include "messageBus.hpp"

using cqueue = moodycamel::ConcurrentQueue<queueMessage>;
using cqueue_ptr = std::shared_ptr<cqueue>;

class BoostMessage : public MessageBus {

private:
  std::unordered_map<std::string, cqueue_ptr> name2Queue;
  std::shared_mutex m;

public:
  BoostMessage();

  virtual bool registered(std::string const &name) override;

  virtual bool unregistered(std::string const &name) override;

  virtual bool send(std::string const &source, std::string const &target,
                    MessageType const &type, queueMessage message) override;

  virtual bool recv(std::string const &name, std::string &sender,
                    MessageType &type, queueMessage &message) override;
};

#endif // FLOWCORE_BOOSTMESSAGE_H
