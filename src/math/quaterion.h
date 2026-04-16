#pragma once

#include "math/matrix.h"

namespace math {

template <typename DType> class Quaternion : public Matrix<DType, 4U, 1U> {
public:
  HOST_DEVICE_FUNC Quaternion()
      : Matrix<DType, 4U, 1U>(static_cast<DType>(1), static_cast<DType>(0),
                              static_cast<DType>(0), static_cast<DType>(0)) {}

  HOST_DEVICE_FUNC Quaternion(DType const w, DType const x, DType const y,
                              DType const z)
      : Matrix<DType, 4U, 1U>(w, x, y, z) {}

  HOST_DEVICE_FUNC Quaternion(Matrix<DType, 3U, 3U> const &rot_mat) {
    DType const trace{rot_mat.Trace()};
    if (trace > static_cast<DType>(0)) {
      double const phi{acosf(0.5 * (trace - 1.0))};
      double const half_phi{0.5 * phi};
      double const cos_half_phi{cosf(half_phi)};
      double const quad_inv_cos_half_phi{0.25 / cos_half_phi};
      this->w() = static_cast<DType>(cos_half_phi);
      this->x() = static_cast<DType>((rot_mat(2, 1) - rot_mat(1, 2)) *
                                     quad_inv_cos_half_phi);
      this->y() = static_cast<DType>(-(rot_mat(2, 0) - rot_mat(0, 2)) *
                                     quad_inv_cos_half_phi);
      this->z() = static_cast<DType>((rot_mat(1, 0) - rot_mat(0, 1)) *
                                     quad_inv_cos_half_phi);
    } else {
      uint32_t i{0U};
      if (rot_mat(1, 1) > rot_mat(0, 0)) {
        i = 1U;
      }
      if (rot_mat(2, 2) > rot_mat(i, i)) {
        i = 2U;
      }

      uint32_t const j{(i + 2U) % 3U};
      uint32_t const k{(i + 1U) % 3U};
      DType const t{
          DType(sqrt(1.0 + rot_mat(i, i) - rot_mat(j, j) - rot_mat(k, k)))};
      this->data_[i] = static_cast<DType>(0.5 * t);
      DType const scale{static_cast<DType>(0.5 / t)};
      this->data_[3] = (rot_mat(j, k) - rot_mat(k, j)) * scale;
      this->data_[k] = (rot_mat(i, k) + rot_mat(k, i)) * scale;
      this->data_[j] = (rot_mat(i, j) + rot_mat(j, i)) * scale;
    }
  }

  HOST_DEVICE_FUNC Matrix<DType, 3U, 3U> ToRotationMatrix() const {
    DType const w2{this->w() * this->w()};
    DType const x2{this->x() * this->x()};
    DType const y2{this->y() * this->y()};
    DType const z2{this->z() * this->z()};
    DType const wx{this->w() * this->x()};
    DType const wy{this->w() * this->y()};
    DType const wz{this->w() * this->z()};
    DType const xy{this->x() * this->y()};
    DType const xz{this->x() * this->z()};
    DType const yz{this->y() * this->z()};
    Matrix<DType, 3U, 3U> rot_mat{};
    rot_mat(0, 0) = w2 + x2 - y2 - z2;
    rot_mat(0, 1) = 2.0 * (xy - wz);
    rot_mat(0, 2) = 2.0 * (xz + wy);
    rot_mat(1, 0) = 2.0 * (xy + wz);
    rot_mat(1, 1) = w2 - x2 + y2 - z2;
    rot_mat(1, 2) = 2.0 * (yz - wx);
    rot_mat(2, 0) = 2.0 * (xz - wy);
    rot_mat(2, 1) = 2.0 * (yz + wx);
    rot_mat(2, 2) = w2 - x2 - y2 + z2;

    return rot_mat;
  }

  HOST_DEVICE_FUNC static Quaternion<DType> Identity() {
    return Quaternion<DType>();
  }

  HOST_DEVICE_FUNC Quaternion<DType> Normalized() const {
    Quaternion<DType> ret{*this};
    DType const norm{this->Norm()};
    if (norm <= 1.0e-32) {
      return ret;
    }

    ret.w() /= norm;
    ret.x() /= norm;
    ret.y() /= norm;
    ret.z() /= norm;

    return ret;
  }

  HOST_DEVICE_FUNC void SetIdentity() {
    this->SetZero();
    this->w() = static_cast<DType>(1);
  }

  HOST_DEVICE_FUNC Quaternion<DType> Conjugated() const {
    return Quaternion<DType>(this->w(), -this->x(), -this->y(), -this->z());
  }

  HOST_DEVICE_FUNC Quaternion<DType>
  operator*(Quaternion<DType> const &other) const {
    return Quaternion<DType>(this->w() * other.w() - this->x() * other.x() -
                                 this->y() * other.y() - this->z() * other.z(),
                             this->x() * other.w() + this->w() * other.x() -
                                 this->z() * other.y() + this->y() * other.z(),
                             this->y() * other.w() + this->z() * other.x() +
                                 this->w() * other.y() - this->x() * other.z(),
                             this->z() * other.w() - this->y() * other.x() +
                                 this->x() * other.y() + this->w() * other.z());
  }

  HOST_DEVICE_FUNC Matrix<DType, 3U, 1U>
  operator*(Matrix<DType, 3U, 1U> const &v) const {
    return ToRotationMatrix() * v;
  }
};

template <typename DType>
std::ostream &operator<<(std::ostream &os, Quaternion<DType> const &q) {
  os << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z();
  return os;
}

typedef Quaternion<float> Quaternionf;
typedef Quaternion<double> Quaterniond;

} // namespace math