#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
int main() {
  MatrixXd v1(1, 16);
  MatrixXd v2(10, 16);

  v1.setRandom();
  v2.setRandom();

  // 计算v1和v2之间的cosine相似度
  MatrixXd norm_v1 = v1.rowwise().norm();
  MatrixXd norm_v2 = v2.rowwise().norm();

  // 将分母中为0的元素置为一个很小的非零值
  norm_v1 = (norm_v1.array() == 0).select(1e-8, norm_v1);
  norm_v2 = (norm_v2.array() == 0).select(1e-8, norm_v2);

  MatrixXd dot_product = v1 * v2.transpose();  // 1x10

  MatrixXd cosine_sim = dot_product.array() / (norm_v1 * norm_v2.transpose()).array();

  // 输出结果
  std::cout << v1 << std::endl;
  std::cout << v2 << std::endl;
  std::cout << cosine_sim << std::endl;

  return 0;
}