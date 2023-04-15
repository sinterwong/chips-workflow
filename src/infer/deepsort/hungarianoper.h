#ifndef __INFER_DEEPSORT_HUNGARIANOPER_H_
#define __INFER_DEEPSORT_HUNGARIANOPER_H_
#include "dataType.hpp"
#include "munkres.h"

class HungarianOper {
public:
  static Eigen::Matrix<float, -1, 2, Eigen::RowMajor>
  Solve(const DYNAMICM &cost_matrix);
};

#endif // HUNGARIANOPER_H