//
// Created by Wallel on 2021/12/27.
//

#include "lightTrack.h"

inline float sigmoid(float x) { return 1 / (1 + std::exp(-x)); }

lightTrack::lightTrack(NNEngine *_engine, const std::vector<std::string> &input,
                       const std::vector<std::string> &output) {
  engine = _engine;
  inputName = input;
  outputName = output;
  grid();
  hanning_windows();
}

tensorList
lightTrack::forward(const std::unordered_map<std::string, cv::Mat> &input) {
  tensorList engineInput = engine->dataProcess(input, mean, scale);
  return engine->forward(engineInput);
}

std::array<float, 4> lightTrack::track(cv::Mat img) {
  float wc_z = target_sz[0] + context_amount * (target_sz[0] + target_sz[1]);
  float hc_z = target_sz[1] + context_amount * (target_sz[0] + target_sz[1]);
  float s_z = std::round(std::sqrt(wc_z * hc_z));

  float scale_z = float(exemplar_size) / s_z;
  float d_search = float(instance_size - exemplar_size) / 2;
  float pad = d_search / scale_z;
  float s_x = s_z + 2 * pad;

  cv::Mat crop_x = getSubWindow_SiamFC(img, target_pos, instance_size,
                                       std::round(s_x), avg_chans);

  std::unordered_map<std::string, cv::Mat> input;
  input.insert({inputName[0], crop_z});
  input.insert({inputName[1], crop_x});

  auto output = forward(input);

  memcpy(pred_cls.get(), (float *)output[outputName[0]],
         sizeof(float) * score_size * score_size);

  float maxScore = -1, maxLocation = -1, maxPenalty;

  for (int i = 0; i < score_size * score_size; i++) {
    pred_x1[i] = grid_x[i] - output.at<float>(outputName[1], i * 4 + 0);
    pred_y1[i] = grid_y[i] - output.at<float>(outputName[1], i * 4 + 1);
    pred_x2[i] = grid_x[i] + output.at<float>(outputName[1], i * 4 + 2);
    pred_y2[i] = grid_y[i] + output.at<float>(outputName[1], i * 4 + 3);

    float s_c = change(sz(pred_x2[i] - pred_x1[i], pred_y2[i] - pred_y1[i]) /
                       (sz(target_sz[0], target_sz[1])));

    float r_c = change((target_sz[0] / target_sz[1]) /
                       ((pred_x2[i] - pred_x1[i]) / (pred_y2[i] - pred_y1[i])));
    float penalty = std::exp(-(r_c * s_c - 1) * penalty_k);
    float pscore = penalty * sigmoid(pred_cls[i]);

    pscore = pscore * (1 - window_influence) + window[i] * window_influence;

    if (pscore > maxScore) {
      maxScore = pscore;
      maxLocation = i;
      maxPenalty = penalty;
    }
  }

  float pred_xs = (pred_x1[maxLocation] + pred_x2[maxLocation]) / 2;
  float pred_ys = (pred_y1[maxLocation] + pred_y2[maxLocation]) / 2;
  float pred_w = pred_x2[maxLocation] - pred_x1[maxLocation];
  float pred_h = pred_y2[maxLocation] - pred_y1[maxLocation];
  float diff_xs = pred_xs - instance_size / 2;
  float diff_ys = pred_ys - instance_size / 2;

  diff_xs /= scale_z;
  diff_ys /= scale_z;
  pred_w /= scale_z;
  pred_h /= scale_z;

  float next_lr = maxPenalty * sigmoid(pred_cls[maxLocation]) * lr;
  float res_xs = target_pos[0] + diff_xs;
  float res_ys = target_pos[1] + diff_ys;
  float res_w = pred_w * next_lr + (1 - next_lr) * target_sz[0];
  float res_h = pred_h * next_lr + (1 - next_lr) * target_sz[1];

  target_pos = {res_xs, res_ys};
  target_sz = {target_sz[0] * (1 - next_lr) + next_lr * res_w,
               target_sz[1] * (1 - next_lr) + next_lr * res_h};

  return {target_pos[0] - target_sz[0] / 2, target_pos[1] - target_sz[1] / 2,
          target_sz[0], target_sz[1]};
}

double lightTrack::change(double r) { return std::max(r, 1. / r); }

double lightTrack::sz(double w, double h) {
  double pad = (w + h) * 0.5;
  double sz2 = (w + pad) * (h + pad);
  return std::sqrt(sz2);
}

void lightTrack::grid() {
  int sz_x = score_size / 2;
  int sz_y = score_size / 2;

  for (int i = 0; i < score_size; i++) {
    for (int j = 0; j < score_size; j++) {
      grid_x[i * score_size + j] = (j - sz_x) * stride + instance_size / 2;
      grid_y[i * score_size + j] = (i - sz_y) * stride + instance_size / 2;
    }
  }
}

void lightTrack::hanning_windows() {
  for (int i = 0; i < score_size; i++) {
    for (int j = 0; j < score_size; j++) {
      window[i * score_size + j] =
          0.25 * (1 - std::cos(float(2 * pi * i) / float(score_size))) *
          (1 - std::cos(float(2 * pi * j) / float(score_size)));
    }
  }
}

void lightTrack::init(cv::Mat img, std::array<float, 4> roi) {
  target_sz = {roi[2], roi[3]};
  target_pos = {roi[0] + roi[2] / 2, roi[1] + roi[3] / 2};
  float wc_z = target_sz[0] + context_amount * (target_sz[0] + target_sz[1]);
  float hc_z = target_sz[1] + context_amount * (target_sz[0] + target_sz[1]);
  float s_z = std::round(std::sqrt(wc_z * hc_z));
  avg_chans = cv::mean(img);
  crop_z = getSubWindow_SiamFC(img, target_pos, exemplar_size, s_z, avg_chans);
}

cv::Mat getSubWindow_SiamFC(cv::Mat im, std::array<float, 2> pos,
                            float model_sz, float original_sz,
                            cv::Scalar avg_chans) {
  float sz = original_sz;
  float c = (original_sz + 1) / 2;
  std::array<int, 2> im_sz{im.rows, im.cols};
  int context_xmin = int(round(pos[0] - c - 1e-4));
  int context_xmax = int(context_xmin + sz - 1);
  int context_ymin = int(round(pos[1] - c + 1e-4));
  int context_ymax = int(context_ymin + sz - 1);

  int left_pad = std::max(0, -context_xmin);
  int top_pad = std::max(0, -context_ymin);
  int right_pad = std::max(0, context_xmax - im_sz[1] + 1);
  int bottom_pad = std::max(0, context_ymax - im_sz[0] + 1);

  int width = context_xmax + 1 < im_sz[1] ? context_xmax - context_xmin + 1
                                          : im_sz[1] - context_xmin;
  int height = context_ymax + 1 < im_sz[0] ? context_ymax - context_ymin + 1
                                           : im_sz[0] - context_ymin;

  context_xmin = std::max(0, context_xmin);
  context_ymin = std::max(0, context_ymin);

  cv::Mat output;

  output = im(cv::Rect(context_xmin, context_ymin, width, height));
  cv::copyMakeBorder(output, output, top_pad, bottom_pad, left_pad, right_pad,
                     cv::BORDER_CONSTANT, avg_chans);
  if (model_sz != original_sz)
    cv::resize(output, output, cv::Size(model_sz, model_sz));
  return output;
}
