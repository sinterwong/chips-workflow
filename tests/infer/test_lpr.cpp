#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>

#include <Eigen/Dense>

#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include "preprocess.hpp"
#include "visionInfer.hpp"

DEFINE_string(img, "", "Specify face1 image path.");
DEFINE_string(det_model_path, "", "Specify the lprDet model path.");
DEFINE_string(rec_model_path, "", "Specify the lprNet model path.");

using namespace infer;
using namespace common;

using algo_ptr = std::shared_ptr<AlgoInfer>;

algo_ptr getVision(AlgoConfig &&config) {

  std::shared_ptr<AlgoInfer> vision = std::make_shared<VisionInfer>(config);
  if (!vision->init()) {
    FLOWENGINE_LOGGER_ERROR("Failed to init vision");
    std::exit(-1); // 强制中断
    return nullptr;
  }
  return vision;
}

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  PointsDetAlgo lprDet_config{{
                                  1,
                                  {"images"},
                                  {"output"},
                                  FLAGS_det_model_path,
                                  "YoloPDet",
                                  {640, 640, 3},
                                  false,
                                  0,
                                  0,
                                  0.3,
                              },
                              4,
                              0.4};
  AlgoConfig pdet_config;
  pdet_config.setParams(lprDet_config);

  ClassAlgo lprNet_config{{
      1,
      {"images"},
      {"output"},
      FLAGS_rec_model_path,
      "CRNN",
      {176, 48, 3},
      false,
      0,
      0,
      0.3,
  }};
  AlgoConfig prec_config;
  prec_config.setParams(lprNet_config);

  auto lprDet = getVision(std::move(pdet_config));
  auto lprNet = getVision(std::move(prec_config));

  gflags::ShutDownCommandLineFlags();

  return 0;
}

/*
./test_face --img1 /root/workspace/softwares/flowengine/data/face1.jpg \
            --img2 /root/workspace/softwares/flowengine/data/face2.jpg \
            --model_path
/root/workspace/softwares/flowengine/models/facenet_mobilenet_160x160.bin
*/