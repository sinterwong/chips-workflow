#ifndef __INFER_DEEPSORT_TRACKER_H_
#define __INFER_DEEPSORT_TRACKER_H_
#include <vector>

#include "kalmanfilter.h"
#include "nearNeighborDisMetric.h"
#include "track.h"

namespace infer::solution {
class DeepSortTracker {
public:
  NearNeighborDisMetric *metric;
  float max_iou_distance;
  int max_age;
  int n_init;

  KalmanFilter *kf;

  int _next_idx;

public:
  std::vector<Track> tracks;
  DeepSortTracker(/*NearNeighborDisMetric* metric,*/
          float max_cosine_distance, int nn_budget,
          float max_iou_distance = 0.7, int max_age = 30, int n_init = 3);
  void predict();
  void update(const DETECTIONS &detections);
  typedef DYNAMICM (DeepSortTracker::*GATED_METRIC_FUNC)(
      std::vector<Track> &tracks, const DETECTIONS &dets,
      const std::vector<int> &track_indices,
      const std::vector<int> &detection_indices);

private:
  void _match(const DETECTIONS &detections, TRACHER_MATCHD &res);
  void _initiate_track(const DETECTION_ROW &detection);

public:
  DYNAMICM gated_matric(std::vector<Track> &tracks, const DETECTIONS &dets,
                        const std::vector<int> &track_indices,
                        const std::vector<int> &detection_indices);
  DYNAMICM iou_cost(std::vector<Track> &tracks, const DETECTIONS &dets,
                    const std::vector<int> &track_indices,
                    const std::vector<int> &detection_indices);
  Eigen::VectorXf iou(DETECTBOX &bbox, DETECTBOXSS &candidates);
};
}
#endif // TRACKER_H