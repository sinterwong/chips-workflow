#include <cstdlib>
#include <gflags/gflags.h>
#include <iostream>

#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/video/tracking.hpp>
#include <set>
#include <unordered_map>
#include <utility>
#include <variant>

#include "dataType.hpp"
#include "visionInfer.hpp"

#include "infer/preprocess.hpp"
#include "module/videoManager.hpp"

#include "infer/tracker.h"

DEFINE_string(url, "", "Specify stream url.");
DEFINE_string(det_model_path, "", "Specify the lprDet model path.");
DEFINE_string(reid_model_path, "", "Specify the lprNet model path.");

using namespace infer;
using namespace infer::solution;
using namespace common;

using algo_ptr = std::shared_ptr<AlgoInfer>;

using RESULT_DATA = std::pair<int, DETECTBOX>;

algo_ptr getVision(AlgoConfig &&config) {

  std::shared_ptr<AlgoInfer> vision = std::make_shared<VisionInfer>(config);
  if (!vision->init()) {
    FLOWENGINE_LOGGER_ERROR("Failed to init vision");
    std::exit(-1); // 强制中断
    return nullptr;
  }
  return vision;
}

void inference(cv::Mat &image, InferResult &ret,
               std::shared_ptr<AlgoInfer> vision) {

  RetBox region{"hello", {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};

  InferParams params{std::string("hello"),
                     ColorType::NV12,
                     0.0,
                     region,
                     {image.cols, image.rows, image.channels()}};

  vision->infer(image.data, params, ret);
}

cv::Rect2i getRect(BBox const &bbox) {
  return cv::Rect2i{static_cast<int>(bbox.bbox[0]),
                    static_cast<int>(bbox.bbox[1]),
                    static_cast<int>(bbox.bbox[2] - bbox.bbox[0]),
                    static_cast<int>(bbox.bbox[3] - bbox.bbox[1])};
}

void get_detections(DETECTBOX box, float confidence, DETECTIONS &d) {
  DETECTION_ROW tmpRow;
  tmpRow.tlwh = box; // DETECTBOX(x, y, w, h);

  tmpRow.confidence = confidence;
  d.push_back(tmpRow);
}

int main(int argc, char **argv) {

  gflags::SetUsageMessage("some usage message");
  gflags::SetVersionString("1.0.0");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 算法启动
  DetAlgo person_det_params{{
                                1,
                                {"images"},
                                {"output"},
                                FLAGS_det_model_path,
                                "Yolo",
                                {640, 640, 3},
                                false,
                                0,
                                0,
                                0.3,
                            },
                            0.4};
  AlgoConfig det_config;
  det_config.setParams(person_det_params);

  FeatureAlgo reid_params{{
                              1,
                              {"images"},
                              {"output"},
                              FLAGS_reid_model_path,
                              "FaceNet",
                              {128, 256, 3},
                              false,
                              0,
                              0,
                              0.3,
                          },
                          512};
  AlgoConfig reid_config;
  reid_config.setParams(reid_params);

  auto personDet = getVision(std::move(det_config));
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");

  auto reidNet = getVision(std::move(reid_config));
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");

  // 视频流
  module::utils::VideoManager vm{FLAGS_url};

  vm.init();
  FLOWENGINE_LOGGER_INFO("Video manager has initialized!");
  vm.run();
  FLOWENGINE_LOGGER_INFO("Video manager is running!");

  // trakcker
  DeepSortTracker deepsrot(0.2, 100);

  std::vector<RESULT_DATA> last_results;
  DETECTIONS last_detections;

  int count = 0;

  std::set<int> counter;
  while (vm.isRunning()) {
    count++;
    auto image = vm.getcvImage();
    cv::Mat show_image;
    cv::cvtColor(*image, show_image, CV_YUV2BGR_NV12);
    std::vector<RESULT_DATA> results;
    DETECTIONS detections;
    // 获取每个目标及其特征
    if (count % 3 == 0) {
      // 行人检测结果
      InferResult detRet;
      inference(*image, detRet, personDet);

      auto bboxes = std::get_if<BBoxes>(&detRet.aRet);
      if (!bboxes) {
        FLOWENGINE_LOGGER_ERROR("Person detection is failed!");
        continue;
      }

      for (auto &bbox : *bboxes) {

        DETECTION_ROW tmpRow;
        tmpRow.tlwh =
            DETECTBOX{bbox.bbox[0], bbox.bbox[1], bbox.bbox[2] - bbox.bbox[0],
                      bbox.bbox[3] - bbox.bbox[1]};
        tmpRow.confidence = bbox.class_confidence;

        cv::Mat cropedImage;
        auto rect = getRect(bbox);
        utils::cropImage(*image, cropedImage, rect, ColorType::NV12);

        InferResult reidRet;
        inference(cropedImage, reidRet, reidNet);
        auto feature = std::get_if<Eigenvector>(&reidRet.aRet);
        if (!feature) {
          FLOWENGINE_LOGGER_ERROR("Feature extract is failed!");
          continue;
        }
        for (size_t i = 0; i < feature->size(); ++i) {
          tmpRow.feature[i] = feature->at(i);
        }

        detections.emplace_back(tmpRow);
      }

      deepsrot.predict();
      deepsrot.update(detections);

      for (Track &track : deepsrot.tracks) {
        if (!track.is_confirmed() || track.time_since_update > 1)
          continue;
        results.push_back(std::make_pair(track.track_id, track.to_tlwh()));
      }
      last_results = results;
      last_detections = detections;
    } else {
      results = last_results;
      detections = last_detections;
    }

    for (size_t j = 0; j < results.size(); j++) {
      counter.insert(results[j].first);
      DETECTBOX tmp = results[j].second;
      cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
      rectangle(show_image, rect, cv::Scalar(255, 255, 0), 2);

      std::string label = cv::format("%d", results[j].first);
      cv::putText(show_image, label, cv::Point(rect.x, rect.y),
                  cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 14, 50), 2);
    }
    std::cout << "person number: " << counter.size() << std::endl;
    cv::imwrite("object_counter_out.jpg", show_image);
  }

  gflags::ShutDownCommandLineFlags();

  return 0;
}

/*
./test_face --img1 /root/workspace/softwares/flowengine/data/face1.jpg \
            --img2 /root/workspace/softwares/flowengine/data/face2.jpg \
            --model_path
/root/workspace/softwares/flowengine/models/facenet_mobilenet_160x160.bin
*/