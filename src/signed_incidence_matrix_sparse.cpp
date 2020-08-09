#include "signed_incidence_matrix_sparse.h"
#include <vector>

void signed_incidence_matrix_sparse(
  const int n,
  const Eigen::MatrixXi & E,
  Eigen::SparseMatrix<double>  & A)
{
  //////////////////////////////////////////////////////////////////////////////
  std::vector<Eigen::Triplet<double> > ijv;
  for (int i = 0; i < E.rows(); i++) {
    for (int j = 0; j < n; j++) {
      if (j == E(i, 0)) ijv.emplace_back(i, j, 1);
      else if (j == E(i, 1)) ijv.emplace_back(i, j, -1);
    }
  }
  A.resize(E.rows(),n);
  A.setFromTriplets(ijv.begin(),ijv.end());
  //////////////////////////////////////////////////////////////////////////////
}
