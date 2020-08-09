#include "fast_mass_springs_precomputation_sparse.h"
#include "signed_incidence_matrix_sparse.h"
#include <vector>

bool fast_mass_springs_precomputation_sparse(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & E,
  const double k,
  const Eigen::VectorXd & m,
  const Eigen::VectorXi & b,
  const double delta_t,
  Eigen::VectorXd & r,
  Eigen::SparseMatrix<double>  & M,
  Eigen::SparseMatrix<double>  & A,
  Eigen::SparseMatrix<double>  & C,
  Eigen::SimplicialLLT<Eigen::SparseMatrix<double> > & prefactorization)
{
  /////////////////////////////////////////////////////////////////////////////
  Eigen::SparseMatrix<double> Q(V.rows(), V.rows());
  r = Eigen::VectorXd(E.rows());
	M.resize(V.rows(),V.rows());
  A.resize(E.rows(), V.rows());
	C.resize(b.size(), V.rows()); 
  std::vector<Eigen::Triplet<double> > ijv;

  for (int i = 0; i < E.rows(); i++) {
    r[i] = (V.row(E(i, 0)) - V.row(E(i, 1))).norm();
  }

  ijv.clear();
  for (int i = 0; i < V.rows(); i++) {
    ijv.emplace_back(i, i, m[i]);
  }
  M.setFromTriplets(ijv.begin(),ijv.end());

  signed_incidence_matrix_sparse(V.rows(), E, A);

  ijv.clear();
  for (int i = 0; i < b.size(); i++) {
    ijv.emplace_back(i, b(i), 1);
  }
  C.setFromTriplets(ijv.begin(),ijv.end());

  const double w = 1e10;
  Q = k * A.transpose() * A + 1 / (delta_t * delta_t) * M + w * C.transpose() * C;
  /////////////////////////////////////////////////////////////////////////////
  prefactorization.compute(Q);
  return prefactorization.info() != Eigen::NumericalIssue;
}
