/**
 * @file assdDet.cpp
 * @author Sinter Wong (sintercver@gmail.com)
 * @brief
 * @version 0.1
 * @date 2022-08-28
 *
 * @copyright Copyright (c) 2022
 *
 */
#include "assdDet.hpp"
#include "opencv2/imgcodecs.hpp"
#include <cassert>
#include <string>

namespace infer {
namespace vision {
void Assd::generateBoxes(std::unordered_map<int, BBoxes> &m,
                         void **outputs) const {

  float **output = static_cast<float **>(*outputs);
  assert(static_cast<int>(modelInfo.outputShapes.size()) ==
         modelInfo.output_count);

  // TODO Re-implement
  for (int i = 0; i < modelInfo.output_count; ++i) {

    int fea_h = modelInfo.outputShapes[i][2];
    int fea_w = modelInfo.outputShapes[i][3];

    /*CHECK score map with opencv*/
    /*
    cv::Mat temp_score_map = cv::Mat(fea_h, fea_w, CV_8UC1);
    for (int y = 0; y < fea_h; y++) {
      for (int x = 0; x < fea_w; x++) {
        int k = y * fea_w + x;
        temp_score_map.at<uchar>(y, x) = static_cast<uchar>(output[i][k] * 255);
      }
    }
    // cv::resize(temp_score_map, temp_score_map, cv::Size(1280, 720));
    cv::imwrite("score_map" + std::to_string(i) + ".jpg", temp_score_map);
    */

    int fea_spacial_size = fea_w * fea_h;
    for (int y = 0; y < fea_h; y++) {
      for (int x = 0; x < fea_w; x++) {
        int k = y * fea_w + x;
        float center_start = receptive_field_center_start[i];
        float stride = receptive_field_stride[i];

        if (output[i][k] > mParams.cond_thr) {
          float RF_center_x = center_start + stride * float(x);
          float RF_center_y = center_start + stride * float(y);
          float x1 =
              RF_center_x - output[i][1 * fea_spacial_size + k] * RF_half[i];

          float y1 =
              RF_center_y - output[i][2 * fea_spacial_size + k] * RF_half[i];

          float x2 =
              RF_center_x - output[i][3 * fea_spacial_size + k] * RF_half[i];

          float y2 =
              RF_center_y - output[i][4 * fea_spacial_size + k] * RF_half[i];

          float re_x1 = std::min(x1, x2);
          float re_y1 = std::min(y1, y2);
          float re_x2 = std::max(x1, x2);
          float re_y2 = std::max(y1, y2);
          re_x1 = re_x1 < 0 ? 0 : re_x1;
          re_y1 = re_y1 < 0 ? 0 : re_y1;
          re_x2 = re_x2 > mParams.inputShape[0] - 1 ? mParams.inputShape[0] - 1
                                                    : re_x2;
          re_y2 = re_y2 > mParams.inputShape[1] - 1 ? mParams.inputShape[1] - 1
                                                    : re_y2;
          float w = re_x2 - re_x1;
          float h = re_y2 - re_y1;
          float cx = re_x2 - w / 2.;
          float cy = re_y2 - h / 2.;
          BBox det;
          det.class_id = 0;
          det.bbox = {cx, cy, w, h};
          det.class_confidence = output[i][k];

          if (m.count(det.class_id) == 0) {
            // 目前还没有该类别，需要初始化一下
            m.emplace(det.class_id, BBoxes());
          }
          m[det.class_id].push_back(det);
        }
      }
    }
  }
}

bool Assd::verifyOutput(InferResult const &result) const { return true; }

FlowEngineModuleRegister(Assd, const common::AlgorithmConfig &,
                         ModelInfo const &);

} // namespace vision
} // namespace infer