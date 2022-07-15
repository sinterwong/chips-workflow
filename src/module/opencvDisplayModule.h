//
// Created by Wallel on 2022/2/22.
//

#ifndef METAENGINE_OPENCVDISPLAYMODULE_H
#define METAENGINE_OPENCVDISPLAYMODULE_H

#include <opencv2/opencv.hpp>
#include "frameMessage.pb.h"
#include "module.hpp"

#define DIRECT true

namespace module {
class OpencvDisplayModule : public Module
{
protected:
    cv::Mat frame;

public:
    OpencvDisplayModule(Backend *ptr,
                        const std::string &initName,
                        const std::string &initType,
                        const std::vector<std::string> &recv = {},
                        const std::vector<std::string> &send = {},
                        const std::vector<std::string> &pool = {});

    void
    forward(std::vector<std::tuple<std::string, std::string, queueMessage>> message) override;
};
}
#endif //METAENGINE_OPENCVDISPLAYMODULE_H
