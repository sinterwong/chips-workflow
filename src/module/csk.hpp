/*******************************************************************************
 * Created by Qiang Wang on 16/7.24
 * Copyright 2016 Qiang Wang.  [wangqiang2015-at-ia.ac.cn]
 * Licensed under the Simplified BSD License
 *******************************************************************************/
#pragma once

//#define DISPLAY

#include "BasicTracker.h"
#include "csk_ffttools.hpp"
#include "csk_recttools.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>

#define TARGET_LEN 32
#define CSK_SEARCHING_RATIO 4
#define CSK_SEARCHING_RATIO_HALF 2
#define CSK_SEARCHING_IMG_LEN TARGET_LEN *CSK_SEARCHING_RATIO

namespace CSK {
class CSKTracker : public Tracker {

public:
  // Constructor
  CSKTracker();

  // Initialize tracker
  virtual void init(cv::Mat image, const cv::Rect &roi);

  virtual std::tuple<std::array<float, 4>, float> update(cv::Mat image);

  float output_sigma_factor;
  float sigma;
  float lambda;

  // fine CSK parameters
  float padding;
  float interp_factor;
  float loss_criteria; // to detect the loss of the target

  // coarse CSK parameters
  float w_padding;
  float w_interp_factor;
  float downsample;

protected:
  // Detect object in the current frame.
  void detect_coarse(cv::Mat tmpl_img, cv::Mat src_img, float &peak_value);

  void detect_fine(cv::Mat tmpl_img, cv::Mat src_img, float &peak_value);

  // train tracker with a single image
  void train(cv::Mat x, float train_interp_factor, bool fine);

  cv::Mat GetBgdWindow(const cv::Mat &frame, cv::Size sz, cv::Size rsz);

  void CalTargetMeanStdDev(const cv::Mat &frame, cv::Size sz);

  cv::Mat GetSubWindow(const cv::Mat &frame, cv::Size sz);

  void DenseGaussKernel(float sigma, const cv::Mat &x, const cv::Mat &y,
                        cv::Mat &k);

  cv::Mat createHanningMats(cv::Size sz);

  cv::Mat CreateGaussian1(int n, float sigma, int ktype);

  cv::Mat CreateGaussian2(cv::Size sz, float sigma, int ktype);

  void CircShift(cv::Mat &x, cv::Size sz);

  void preprocess(cv::Mat &input, cv::Mat &output);

  inline cv::Size scale_size(const cv::Size &r, float scale) {
    float width = float(r.width) * scale;
    float height = float(r.height) * scale;
    return cv::Size(cvFloor(width), cvFloor(height));
  }

  cv::Mat _alphaf;
  cv::Mat _prob;
  cv::Mat _tmpl;
  cv::Mat _w_alphaf;
  cv::Mat _w_prob;
  cv::Mat _w_tmpl;
  float _mean;
  float _dev;

  cv::Point2f _pos;

  cv::Rect_<float> _roi_r; // real ROI
  float target_ratio_x, target_ratio_y;
  float target_len;
  int target_roi_x, target_roi_y;

private:
  cv::Mat hann;
  cv::Mat w_hann;
  cv::Size target_sz;
  cv::Size sz;
  cv::Size w_sz;
  cv::Size w_rsz;
  float t_mean;
  float t_dev;
  // avergae peak value;
  float _avg_peak;
  // store the latest 50 peak values;
  std::queue<float> _peak;
};
} // namespace CSK