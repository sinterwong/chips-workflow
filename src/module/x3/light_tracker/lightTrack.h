//
// Created by Wallel on 2021/12/27.
//

#ifndef LIGHTTRACK_MNN_LIGHTTRACK_H
#define LIGHTTRACK_MNN_LIGHTTRACK_H

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference.h"

cv::Mat getSubWindow_SiamFC(cv::Mat im, std::array<float, 2> pos,
                            float model_sz, float original_sz,
                            cv::Scalar avg_chans);

class lightTrack {
protected:
  NNEngine *engine;
  std::vector<std::string> inputName;
  std::vector<std::string> outputName;

  /*
   * HyperParameter
   * */
  std::array<float, 3> mean{123.675f, 116.28f, 103.53f};
  float scale = 0.017124f;

  float penalty_k = 0.062;
  float window_influence = 0.38;
  float lr = 0.765;

  int stride = 16;
  int instance_size = 256, exemplar_size = 127;
  int score_size = instance_size / stride;
  float context_amount = 0.5;
  const float pi = 3.1415926;

  // parameter end

  std::array<float, 2> target_sz, target_pos;
  std::unique_ptr<float[]>
      grid_x = std::make_unique<float[]>(score_size * score_size),
      grid_y = std::make_unique<float[]>(score_size * score_size),
      window = std::make_unique<float[]>(score_size * score_size),
      pred_x1 = std::make_unique<float[]>(score_size * score_size),
      pred_x2 = std::make_unique<float[]>(score_size * score_size),
      pred_y1 = std::make_unique<float[]>(score_size * score_size),
      pred_y2 = std::make_unique<float[]>(score_size * score_size),
      pred_cls = std::make_unique<float[]>(score_size * score_size);
  cv::Scalar avg_chans;
  cv::Mat crop_z;

  void grid();

  double change(double r);

  double sz(double w, double h);

  void hanning_windows();

  tensorList forward(const std::unordered_map<std::string, cv::Mat> &input);

public:
  lightTrack(NNEngine *_engine,
             const std::vector<std::string> &input = {"z", "x"},
             const std::vector<std::string> &output = {"cls", "reg"});

  void init(cv::Mat img, std::array<float, 4> roi);

  std::array<float, 4> track(cv::Mat img);
};

#endif // LIGHTTRACK_MNN_LIGHTTRACK_H
