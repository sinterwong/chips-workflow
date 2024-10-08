#ifndef __INFER_DEEPSORT_LINEAR_ASSIGNMENT_H_
#define __INFER_DEEPSORT_LINEAR_ASSIGNMENT_H_
#include "dataType.hpp"
#include "tracker.h"
#define INFTY_COST 1e5
namespace infer::solution {
// for matching;
class linear_assignment {
  linear_assignment();
  linear_assignment(const linear_assignment &);
  linear_assignment &operator=(const linear_assignment &);
  static linear_assignment *instance;

public:
  static linear_assignment *getInstance();
  TRACHER_MATCHD matching_cascade(
      DeepSortTracker *distance_metric, DeepSortTracker::GATED_METRIC_FUNC distance_metric_func,
      float max_distance, int cascade_depth, std::vector<Track> &tracks,
      const DETECTIONS &detections, std::vector<int> &track_indices,
      std::vector<int> detection_indices = std::vector<int>());
  TRACHER_MATCHD min_cost_matching(
      DeepSortTracker *distance_metric, DeepSortTracker::GATED_METRIC_FUNC distance_metric_func,
      float max_distance, std::vector<Track> &tracks,
      const DETECTIONS &detections, std::vector<int> &track_indices,
      std::vector<int> &detection_indices);
  DYNAMICM gate_cost_matrix(KalmanFilter *kf, DYNAMICM &cost_matrix,
                            std::vector<Track> &tracks,
                            const DETECTIONS &detections,
                            const std::vector<int> &track_indices,
                            const std::vector<int> &detection_indices,
                            float gated_cost = INFTY_COST,
                            bool only_position = false);
};
}
#endif // LINEAR_ASSIGNMENT_H