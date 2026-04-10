#pragma once

#include "math/matrix.h"

namespace math {

template <typename DType, uint32_t N> class LinearSolverBase {
  static_assert(N > 0U);

public:
  LinearSolverBase(Matrix<DType, N, N> const &m){};
  virtual ~LinearSolverBase() = default;

  virtual Matrix<DType, N, 1> Solve(Matrix<DType, N, 1> const &b) = 0;

  virtual void SolveInPlace(Matrix<DType, N, 1> &b) = 0;
};

template <typename DType, uint32_t N>
class CholeskyLinearSolver : public LinearSolverBase<DType, N> {
public:
  explicit CholeskyLinearSolver(Matrix<DType, N, N> const &m)
      : LinearSolverBase<DType, N>(m) {
    Decompose(m);
  }

  Matrix<DType, N, 1> Solve(Matrix<DType, N, 1> const &b) override {
    Matrix<DType, N, 1> x{};
    //* Solve for Lz = b
    for (uint32_t i{0U}; i < N; ++i) {
      x(i) = b(i);
      for (uint32_t j{0U}; j < i; ++j) {
        x(i) -= l_mat_(i, j) * x(j);
      }
    }
    //* Solve for Dy = z
    for (uint32_t i{0U}; i < N; ++i) {
      x(i) /= d_vec_(i);
    }
    //* Solve for L^Tx = y
    for (uint32_t i{0U}; i < N; ++i) {
      for (uint32_t j{0}; j < i; ++j) {
        x(N - i - 1U) -= l_mat_(N - j - 1U, N - i - 1U) * x(N - j - 1U);
      }
    }

    return x;
  }

  void SolveInPlace(Matrix<DType, N, 1> &b) override {
    //* Solve for Lz = b
    for (uint32_t i{0U}; i < N; ++i) {
      for (uint32_t j{0U}; j < i; ++j) {
        b(i) -= l_mat_(i, j) * b(j);
      }
    }
    //* Solve for Dy = z
    for (uint32_t i{0U}; i < N; ++i) {
      b(i) /= d_vec_(i);
    }
    //* Solve for L^Tx = y
    for (uint32_t i{0U}; i < N; ++i) {
      for (uint32_t j{0}; j < i; ++j) {
        b(N - i - 1U) -= l_mat_(N - j - 1U, N - i - 1U) * b(N - j - 1U);
      }
    }
  }

  void Decompose(Matrix<DType, N, N> const &m) {
    DType constexpr kEps{
        static_cast<DType>(std::numeric_limits<DType>::epsilon())};
    DType constexpr kOne{static_cast<DType>(1)};
    valid_ = false;
    for (uint32_t i{0U}; i < N; ++i) {
      l_mat_(i, i) = kOne;
      
      DType d{m(i, i)};
      for (uint32_t k{0U}; k < i; ++k) {
        d -= l_mat_(i, k) * l_mat_(i, k) * d_vec_(k);
      }
      if (d <= kEps) {
        return;
      }
      d_vec_(i) = d;

      for (uint32_t j{i + 1U}; j < N; ++j) {
        l_mat_(j, i) = m(j, i);
        for (uint32_t k{0U}; k < i; ++k) {
          l_mat_(j, i) -= l_mat_(j, k) * l_mat_(i, k) * d_vec_(k);
        }
        l_mat_(j, i) /= d;
      }
    }

    valid_ = true;
  }

  bool valid() const { return valid_; }

  Matrix<DType, N, N> const &l_mat() const { return l_mat_; }

  Matrix<DType, N, 1> const &d_vec() const { return d_vec_; }

protected:
  Matrix<DType, N, N> l_mat_{};
  Matrix<DType, N, 1> d_vec_{};
  bool valid_{false};
};

} // namespace math