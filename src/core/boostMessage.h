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
#include <unordered_map>
#include <vector>

#include "concurrentqueue.h"
#include "messageBus.h"

class BoostMessage : public MessageBus {
protected:
  std::unordered_map<std::string, int> name2Queue;
  std::vector<std::shared_ptr<moodycamel::ConcurrentQueue<queueMessage>>>
      socketRecv;

public:
  BoostMessage();

  virtual bool registered(std::string name) override;

  virtual bool send(std::string source, std::string target, std::string type,
                    queueMessage message) override;

  /*
   * structured Binding (C++ 17) for tuple with string has unknown problem
   * about double free of ptr.
   * So it also use a traditional way to pass the result.
   * */
  virtual bool recv(std::string source, returnFlag &flag, std::string &send,
                    std::string &type, queueMessage &byte,
                    bool waitFlag = true) override;
};

#endif // FLOWCORE_BOOSTMESSAGE_H
