//
// Created by Wallel on 2021/12/26.
//

#ifndef DETTRACKENGINE_MESSAGEBUS_H
#define DETTRACKENGINE_MESSAGEBUS_H

#include <array>
#include <atomic>
#include <iostream>
#include <map>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

// #include <boost/pool/object_pool.hpp>
// #include <boost/pool/pool.hpp>

// #include "basicMessage.pb.h"

struct ResultMessage {
  std::vector<std::pair<std::string, std::array<float, 5>>> bboxes;  // [x1, y1, x2, y2, confidence, classid]
  std::vector<std::pair<std::string, std::array<float, 8>>> polys;  // [x1, y1, ..., x4, y4, confidence, classid]
};

struct queueMessage {
  int width;
  int height;
  int key;
  std::string send;
  std::string recv;
  std::string type;
  std::string str;
  ResultMessage results;
};

class MessageBus {
protected:
  std::unordered_set<std::string> pool;

public:
  enum returnFlag {
    null,
    mapNotFind,
    successWithEmpty,
    successWithMore,
  };

  virtual bool registered(std::string name,
                          const std::vector<std::string> &reqRecvName,
                          const std::vector<std::string> &proRecvName) = 0;

  virtual bool send(std::string source, std::string target, std::string type,
                    queueMessage message) = 0;

  virtual bool recv(std::string source, returnFlag &flag, std::string &send,
                    std::string &type, queueMessage &byte,
                    bool waitFlag = true) = 0;
};

#endif // DETTRACKENGINE_MESSAGEBUS_H
