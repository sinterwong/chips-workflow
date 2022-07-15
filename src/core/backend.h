//
// Created by Wallel on 2022/2/1.
//

#ifndef FLOWCORE_BACKEND_H
#define FLOWCORE_BACKEND_H

#include "messageBus.h"
#include "routeFramePool.h"
#include <string>
#include <vector>

class Backend {
public:
  MessageBus *message;
  RouteFramePool pool;

  Backend(MessageBus *);
};

#endif // FLOWCORE_BACKEND_H
