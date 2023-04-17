#ifndef __INFER_DEEPSORT_DATATYPE_H_
#define __INFER_DEEPSORT_DATATYPE_H_

#include <Eigen/Dense>

namespace infer::solution {

constexpr int FDIMS = 512; // feature dim

using DETECTBOX = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using DETECTBOXSS = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;
using FEATURE = Eigen::Matrix<float, 1, FDIMS, Eigen::RowMajor>;
using FEATURESS = Eigen::Matrix<float, Eigen::Dynamic, FDIMS, Eigen::RowMajor>;

// Kalmanfilter
// typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_FILTER;
using KAL_MEAN = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
using KAL_COVA = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;
using KAL_HMEAN = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using KAL_HCOVA = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

// tracker
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;
struct TRACHER_MATCHD {
  std::vector<MATCH_DATA> matches;
  std::vector<int> unmatched_tracks;
  std::vector<int> unmatched_detections;
};

// linear_assignment
using DYNAMICM = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;

const float kRatio = 0.5;
enum DETECTBOX_IDX { IDX_X = 0, IDX_Y, IDX_W, IDX_H };

class DETECTION_ROW {
public:
  DETECTBOX tlwh;
  float confidence;
  FEATURE feature;
  DETECTBOX to_xyah() const {
    DETECTBOX ret = tlwh;
    ret(0, IDX_X) += (ret(0, IDX_W) * kRatio);
    ret(0, IDX_Y) += (ret(0, IDX_H) * kRatio);
    ret(0, IDX_W) /= ret(0, IDX_H);
    return ret;
  }
  DETECTBOX to_tlbr() const {
    { //(x,y,xx,yy)
      DETECTBOX ret = tlwh;
      ret(0, IDX_X) += ret(0, IDX_W);
      ret(0, IDX_Y) += ret(0, IDX_H);
      return ret;
    }
  }
};
using DETECTIONS = std::vector<DETECTION_ROW>;
} // namespace infer::solution
#endif