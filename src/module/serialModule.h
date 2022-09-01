//
// Created by Wallel on 2022/3/10.
//

#ifndef METAENGINE_SERIALMODULE_H
#define METAENGINE_SERIALMODULE_H

#include <cstring>

#include "frameMessage.pb.h"
#include "module.hpp"
#include "backend.h"
namespace module {
class serialModule : public Module
{
public:
    serialModule(Backend *ptr,
                 const std::string &initName,
                 const std::string &initType,
                 const std::vector <std::string> &recv = {},
                 const std::vector <std::string> &send = {});
    void
    forward(std::vector<std::tuple<std::string, std::string, queueMessage>> message) override;
};

}
#endif //METAENGINE_SERIALMODULE_H
