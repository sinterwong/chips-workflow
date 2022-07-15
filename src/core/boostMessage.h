//
// Created by Wallel on 2022/1/30.
//

#ifndef FLOWCORE_BOOSTMESSAGE_H
#define FLOWCORE_BOOSTMESSAGE_H

#include <memory>
#include <unordered_map>
#include <vector>

// #include <boost/pool/pool.hpp>
#include <condition_variable>

// #include "concurrentqueue.h"
#include "blockingconcurrentqueue.h"
#include "concurrentqueue.h"
#include "messageBus.h"

class BoostMessage : public MessageBus {
protected:
  std::unordered_map<std::string, int> name2Queue;
  std::vector<std::shared_ptr<moodycamel::BlockingConcurrentQueue<queueMessage>>> socketRecv;
  // std::vector<std::shared_ptr<moodycamel::ConcurrentQueue<queueMessage>>> socketRecv;

public:
  BoostMessage();

  bool registered(std::string name, const std::vector<std::string> &reqRecvName,
                  const std::vector<std::string> &proRecvName) override;

  bool send(std::string source, std::string target, std::string type,
            queueMessage message) override;

  /*
   * structured Binding (C++ 17) for tuple with string has unknown problem
   * about double free of ptr.
   * So it also use a traditional way to pass the result.
   * */
  bool recv(std::string source, returnFlag &flag, std::string &send,
            std::string &type, queueMessage &byte,
            bool waitFlag = true) override;
};

#endif // FLOWCORE_BOOSTMESSAGE_H
