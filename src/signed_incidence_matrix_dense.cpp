#include "signed_incidence_matrix_dense.h"

void signed_incidence_matrix_dense(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::MatrixXd & A)
{
  //////////////////////////////////////////////////////////////////////////////
  A = Eigen::MatrixXd::Zero(E.rows(), n);
  for (int i = 0; i < E.rows(); i++) {
    for (int j = 0; j < n; j++) {
      if (j == E(i, 0)) A(i, j) = 1;
      else if (j == E(i, 1)) A(i, j) = -1;
    }
  }
  //////////////////////////////////////////////////////////////////////////////
}
