#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string.h>
#include <type_traits>
#include <vector>

#include "macros.h"

namespace math {

template <typename DType, uint32_t M, uint32_t N> class Matrix {
  static_assert(M > 0 && N > 0, "Size of a matrix must be positive. ");

public:
  HOST_DEVICE_FUNC Matrix() { SetZero(); }

  HOST_DEVICE_FUNC Matrix(DType const x) noexcept { data_[0] = x; }

  HOST_DEVICE_FUNC Matrix(DType const x, DType const y) noexcept {
    static_assert(num_elems_ >= 2U, "");
    data_[0] = x;
    data_[1] = y;
  }

  HOST_DEVICE_FUNC Matrix(DType const x, DType const y,
                          DType const z) noexcept {
    static_assert(num_elems_ >= 3U, "");
    data_[0] = x;
    data_[1] = y;
    data_[2] = z;
  }

  HOST_DEVICE_FUNC Matrix(DType const w, DType const x, DType const y,
                          DType const z) noexcept {
    static_assert(num_elems_ >= 4U, "");
    data_[3] = w;
    data_[0] = x;
    data_[1] = y;
    data_[2] = z;
  }

  HOST_DEVICE_FUNC static Matrix<DType, M, N> Zero() noexcept {
    return Matrix<DType, M, N>();
  }

  HOST_DEVICE_FUNC static Matrix<DType, M, N> Identity() noexcept {
    return Matrix<DType, M, N>().SetIdentity();
  }

  HOST_DEVICE_FUNC static Matrix<DType, M, N> UnitX() noexcept {
    static_assert((M == 1U && N >= 1U) || (M >= 1U && N == 1U));
    return Matrix<DType, M, N>(static_cast<DType>(1));
  }

  HOST_DEVICE_FUNC static Matrix<DType, M, N> UnitY() noexcept {
    static_assert((M == 1U && N >= 2U) || (M >= 2U && N == 1U));
    return Matrix<DType, M, N>(static_cast<DType>(0), static_cast<DType>(1));
  }

  HOST_DEVICE_FUNC static Matrix<DType, M, N> UnitZ() noexcept {
    static_assert((M == 1U && N >= 3U) || (M >= 3U && N == 1U));
    return Matrix<DType, M, N>(static_cast<DType>(0), static_cast<DType>(0),
                               static_cast<DType>(1));
  }

  HOST_DEVICE_FUNC DType *data() noexcept { return data_; }

  HOST_DEVICE_FUNC DType const *data() const noexcept { return data_; }

  HOST_DEVICE_FUNC DType &operator()(uint32_t const i,
                                     uint32_t const j) noexcept {
    return data_[i * cols_ + j];
  }

  HOST_DEVICE_FUNC DType const &operator()(uint32_t const i,
                                           uint32_t const j) const noexcept {
    return data_[i * cols_ + j];
  }

  HOST_DEVICE_FUNC DType &operator()(uint32_t const i) noexcept {
    return data_[i];
  }

  HOST_DEVICE_FUNC DType const &operator()(uint32_t const i) const noexcept {
    return data_[i];
  }

  HOST_DEVICE_FUNC DType &x() noexcept {
    static_assert(rows_ == 1U || cols_ == 1U,
                  "x() is only available for vectors. ");
    return data_[0];
  };

  HOST_DEVICE_FUNC DType &y() noexcept {
    static_assert(rows_ == 1U || cols_ == 1U,
                  "y() is only available for vectors. ");
    return data_[1];
  };

  HOST_DEVICE_FUNC DType &z() noexcept {
    static_assert(rows_ == 1U || cols_ == 1U,
                  "z() is only available for vectors. ");
    return data_[2];
  };

  HOST_DEVICE_FUNC DType &w() noexcept {
    static_assert(rows_ == 1U || cols_ == 1U,
                  "w() is only available for vectors. ");
    return data_[3];
  };

  HOST_DEVICE_FUNC DType const &x() const noexcept {
    static_assert(rows_ == 1U || cols_ == 1U,
                  "x() is only available for vectors. ");
    return data_[0];
  };

  HOST_DEVICE_FUNC DType const &y() const noexcept {
    static_assert(rows_ == 1U || cols_ == 1U,
                  "y() is only available for vectors. ");
    return data_[1];
  };

  HOST_DEVICE_FUNC DType const &z() const noexcept {
    static_assert(rows_ == 1U || cols_ == 1U,
                  "z() is only available for vectors. ");
    return data_[2];
  };

  HOST_DEVICE_FUNC DType const &w() const noexcept {
    static_assert(rows_ == 1U || cols_ == 1U,
                  "w() is only available for vectors. ");
    return data_[3];
  };

  HOST_DEVICE_FUNC Matrix<DType, N, M> Transpose() const {
    Matrix<DType, N, M> ret{};
    for (uint32_t i{0U}; i < M; ++i) {
      for (uint32_t j{0U}; j < N; ++j) {
        ret(j, i) = this->operator()(i, j);
      }
    }

    return ret;
  }

  HOST_DEVICE_FUNC DType Norm() const {
    DType result{static_cast<DType>(0)};
    for (uint32_t i{0U}; i < num_elems_; ++i) {
      result = hypot(result, operator()(i));
    }

    return result;
  }

  HOST_DEVICE_FUNC DType SquaredNorm() const {
    DType result{static_cast<DType>(0)};
    for (uint32_t i{0U}; i < num_elems_; ++i) {
      result += operator()(i) * operator()(i);
    }

    return result;
  }

  HOST_DEVICE_FUNC void Normalize() {
    DType const norm{Norm()};
    if (norm <= 1.0e-32) {
      return;
    }
    *this = *this / norm;
  }

  HOST_DEVICE_FUNC Matrix<DType, M, N> Normalized() const {
    DType const norm{Norm()};
    if (norm <= 1.0e-32) {
      return *this;
    }

    return *this / norm;
  }

  HOST_DEVICE_FUNC void SetZero() {
    memset((void *)data_, 0, sizeof(DType) * num_elems_);
  }

  HOST_DEVICE_FUNC void SetIdentity() {
    SetZero();
    for (uint32_t i{0U}; i < fmin(M, N); ++i) {
      operator()(i, i) = static_cast<DType>(1);
    }
  }

  HOST_DEVICE_FUNC DType Trace() const {
    DType trace{static_cast<DType>(0)};
    for (uint32_t i{0U}; i < fmin(M, N); ++i) {
      trace += operator()(i, i);
    }

    return trace;
  }

  HOST_DEVICE_FUNC DType Dot(Matrix<DType, M, N> const &other) const {
    DType dot{static_cast<DType>(0)};
    for (uint32_t i{0U}; i < num_elems_; ++i) {
      dot += this->operator()(i) * other(i);
    }

    return dot;
  }

  HOST_DEVICE_FUNC Matrix<DType, M, N> &
  operator+=(Matrix<DType, M, N> const &other) {
    for (uint32_t i{0U}; i < num_elems_; ++i) {
      this->operator()(i) += other(i);
    }

    return *this;
  }

  HOST_DEVICE_FUNC Matrix<DType, M, N> &
  operator-=(Matrix<DType, M, N> const &other) {
    for (uint32_t i{0U}; i < num_elems_; ++i) {
      this->operator()(i) -= other(i);
    }

    return *this;
  }

  HOST_DEVICE_FUNC Matrix<DType, M, N> &operator*=(DType const &scalar) {
    for (uint32_t i{0U}; i < num_elems_; ++i) {
      this->operator()(i) *= scalar;
    }

    return *this;
  }

  HOST_DEVICE_FUNC Matrix<DType, M, N> &operator/=(DType const &scalar) {
    for (uint32_t i{0U}; i < num_elems_; ++i) {
      this->operator()(i) /= scalar;
    }

    return *this;
  }

  HOST_DEVICE_FUNC Matrix<DType, M, N>
  CwiseProduct(Matrix<DType, M, N> const &other) const {
    Matrix<DType, M, N> ret{};
    for (uint32_t i{0U}; i < num_elems_; ++i) {
      ret(i) = this->operator()(i) * other(i);
    }

    return ret;
  }

  HOST_DEVICE_FUNC Matrix<DType, M, N>
  Cross(Matrix<DType, M, N> const &other) const {
    static_assert((M == 3U && N == 1U) || (M == 1U && N == 3U));
    Matrix<DType, M, N> cross{};
    cross.x() = -z() * other.y() + y() * other.z();
    cross.y() = z() * other.x() - x() * other.z();
    cross.z() = -y() * other.x() + x() * other.y();

    return cross;
  }

protected:
  DType data_[M * N]{static_cast<DType>(0)};
  static uint32_t constexpr rows_{M};
  static uint32_t constexpr cols_{N};
  static uint32_t const num_elems_{M * N};
};

template <typename DType, uint32_t M, uint32_t N>
HOST_DEVICE_FUNC Matrix<DType, M, N> operator+(Matrix<DType, M, N> const &m0,
                                               Matrix<DType, M, N> const &m1) {
  Matrix<DType, M, N> res{};
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t j{0U}; j < N; ++j) {
      res(i, j) = m0(i, j) + m1(i, j);
    }
  }

  return res;
}

template <typename DType, uint32_t M, uint32_t N>
HOST_DEVICE_FUNC Matrix<DType, M, N> operator+(Matrix<DType, M, N> const &m0) {
  Matrix<DType, M, N> res{};
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t j{0U}; j < N; ++j) {
      res(i, j) = m0(i, j);
    }
  }

  return res;
}

template <typename DType, uint32_t M, uint32_t N>
HOST_DEVICE_FUNC Matrix<DType, M, N> operator-(Matrix<DType, M, N> const &m0,
                                               Matrix<DType, M, N> const &m1) {
  Matrix<DType, M, N> res{};
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t j{0U}; j < N; ++j) {
      res(i, j) = m0(i, j) - m1(i, j);
    }
  }

  return res;
}

template <typename DType, uint32_t M, uint32_t N>
HOST_DEVICE_FUNC Matrix<DType, M, N> operator-(Matrix<DType, M, N> const &m0) {
  Matrix<DType, M, N> res{};
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t j{0U}; j < N; ++j) {
      res(i, j) = -m0(i, j);
    }
  }

  return res;
}

template <typename DType, uint32_t M, uint32_t N>
HOST_DEVICE_FUNC Matrix<DType, M, N> operator*(Matrix<DType, M, N> const &m0,
                                               DType const &scalar) {
  Matrix<DType, M, N> res{};
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t j{0U}; j < N; ++j) {
      res(i, j) = m0(i, j) * scalar;
    }
  }

  return res;
}

template <typename DType, uint32_t M, uint32_t N>
HOST_DEVICE_FUNC Matrix<DType, M, N> operator*(DType const &scalar,
                                               Matrix<DType, M, N> const &m0) {
  Matrix<DType, M, N> res{};
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t j{0U}; j < N; ++j) {
      res(i, j) = m0(i, j) * scalar;
    }
  }

  return res;
}

template <typename DType, uint32_t M, uint32_t N>
HOST_DEVICE_FUNC Matrix<DType, M, N> operator/(Matrix<DType, M, N> const &m0,
                                               DType const &scalar) {
  Matrix<DType, M, N> res{};
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t j{0U}; j < N; ++j) {
      res(i, j) = m0(i, j) / scalar;
    }
  }

  return res;
}

template <typename DType, uint32_t M, uint32_t N, uint32_t P>
HOST_DEVICE_FUNC Matrix<DType, M, P> operator*(Matrix<DType, M, N> const &m0,
                                               Matrix<DType, N, P> const &m1) {
  Matrix<DType, M, P> res{};
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t k{0U}; k < P; ++k) {
      for (uint32_t j{0U}; j < N; ++j) {
        res(i, k) += m0(i, j) * m1(j, k);
      }
    }
  }

  return res;
}

template <typename DType, uint32_t M, uint32_t N>
std::ostream &operator<<(std::ostream &os, Matrix<DType, M, N> const &m) {
  for (uint32_t i{0U}; i < M; ++i) {
    for (uint32_t j{0U}; j < N; ++j) {
      os << std::left << std::setw(8) << m(i, j) << (j < N - 1 ? ", " : "");
    }
    if (i < M - 1) {
      os << "\n";
    }
  }

  return os;
}

typedef Matrix<int32_t, 2, 1> Vector2i;
typedef Matrix<float, 2, 1> Vector2f;
typedef Matrix<double, 2, 1> Vector2d;
typedef Matrix<int32_t, 3, 1> Vector3i;
typedef Matrix<float, 3, 1> Vector3f;
typedef Matrix<double, 3, 1> Vector3d;
typedef Matrix<float, 3, 3> Matrix3f;
typedef Matrix<double, 3, 3> Matrix3d;
typedef Matrix<float, 6, 1> Vector6f;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<float, 6, 6> Matrix6f;
typedef Matrix<double, 6, 6> Matrix6d;

} // namespace math