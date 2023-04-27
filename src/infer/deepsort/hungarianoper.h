#ifndef __INFER_DEEPSORT_HUNGARIANOPER_H_
#define __INFER_DEEPSORT_HUNGARIANOPER_H_
#include "dataType.hpp"
#include "munkres.h"
namespace infer::solution {
class HungarianOper {
public:
  static Eigen::Matrix<float, -1, 2, Eigen::RowMajor>
  Solve(const DYNAMICM &cost_matrix);
};
} // namespace infer::solution

#endif // HUNGARIANOPER_H