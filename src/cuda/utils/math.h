#pragma once

#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdint.h>

namespace cuda {
namespace utils {

template <typename DType>
INLINE_HOST_DEVICE_FUNC void cross_product(DType const *const v0,
                                                       DType const *const v1,
                                                       DType *const result) {

  if (result != v0 && result != v1) {
    result[0] = -v0[2] * v1[1] + v0[1] * v1[2];
    result[1] = v0[2] * v1[0] - v0[0] * v1[2];
    result[2] = -v0[1] * v1[0] + v0[0] * v1[1];
  } else {
    float tmp[3]{};
    tmp[0] = -v0[2] * v1[1] + v0[1] * v1[2];
    tmp[1] = v0[2] * v1[0] - v0[0] * v1[2];
    tmp[2] = -v0[1] * v1[0] + v0[0] * v1[1];
    result[0] = tmp[0];
    result[1] = tmp[1];
    result[2] = tmp[2];
  }
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC DType dot_product(DType const *const v0,
                                                      DType const *const v1) {
  return v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2];
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
vec_add_3d(DType const *const v0, DType const *const v1, DType *const result) {
  result[0] = v0[0] + v1[0];
  result[1] = v0[1] + v1[1];
  result[2] = v0[2] + v1[2];
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
vec_sub_3d(DType const *const v0, DType const *const v1, DType *const result) {
  result[0] = v0[0] - v1[0];
  result[1] = v0[1] - v1[1];
  result[2] = v0[2] - v1[2];
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
vec_scalar_mul_3d(DType const *const v, DType const scalar,
                  DType *const result) {
  result[0] = scalar * v[0];
  result[1] = scalar * v[1];
  result[2] = scalar * v[2];
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
vec_scalar_div_3d(DType const *const v, DType const scalar,
                  DType *const result) {
  result[0] = v[0] / scalar;
  result[1] = v[1] / scalar;
  result[2] = v[2] / scalar;
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC DType norm_3d(DType const *const v) {
  return static_cast<DType>(
      hypotf(hypotf(static_cast<float>(v[0]), static_cast<float>(v[1])),
             static_cast<float>(v[2])));
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void normalize_3d(DType const *const v,
                                                      DType *const nv) {
  float const norm{norm_3d(v)};
  if (norm < 1.0e-10) {
    nv[0] = v[0];
    nv[1] = v[1];
    nv[2] = v[2];
  } else {
    nv[0] = v[0] / norm;
    nv[1] = v[1] / norm;
    nv[2] = v[2] / norm;
  }
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void rotate_3d(DType const *const rot_mat,
                                                   DType const *const v,
                                                   DType *const result) {
  if (result != v) {
    result[0] = rot_mat[0] * v[0] + rot_mat[1] * v[1] + rot_mat[2] * v[2];
    result[1] = rot_mat[3] * v[0] + rot_mat[4] * v[1] + rot_mat[5] * v[2];
    result[2] = rot_mat[6] * v[0] + rot_mat[7] * v[1] + rot_mat[8] * v[2];
  } else {
    DType tmp[3]{};
    tmp[0] = rot_mat[0] * v[0] + rot_mat[1] * v[1] + rot_mat[2] * v[2];
    tmp[1] = rot_mat[3] * v[0] + rot_mat[4] * v[1] + rot_mat[5] * v[2];
    tmp[2] = rot_mat[6] * v[0] + rot_mat[7] * v[1] + rot_mat[8] * v[2];
    result[0] = tmp[0];
    result[1] = tmp[1];
    result[2] = tmp[2];
  }
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
transform_3d(DType const *const transform, DType const *const v,
             DType *const result) {
  if (result != v) {
    result[0] = transform[0] * v[0] + transform[1] * v[1] +
                transform[2] * v[2] + transform[3];
    result[1] = transform[4] * v[0] + transform[5] * v[1] +
                transform[6] * v[2] + transform[7];
    result[2] = transform[8] * v[0] + transform[9] * v[1] +
                transform[10] * v[2] + transform[11];
  } else {
    DType tmp[3]{};
    tmp[0] = transform[0] * v[0] + transform[1] * v[1] + transform[2] * v[2] +
             transform[3];
    tmp[1] = transform[4] * v[0] + transform[5] * v[1] + transform[6] * v[2] +
             transform[7];
    tmp[2] = transform[8] * v[0] + transform[9] * v[1] + transform[10] * v[2] +
             transform[11];
    result[0] = tmp[0];
    result[1] = tmp[1];
    result[2] = tmp[2];
  }
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
transform_3d(DType const *const rot_mat, DType const *const trans_vec,
             DType const *const v, DType *const result) {
  rotate_3d(rot_mat, v, result);
  vec_add_3d(result, trans_vec, result);
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
get_inverse_transform(DType const *const transform,
                      DType const *const inverse) {
  if (inverse != transform) {
    inverse[0] = transform[0];
    inverse[5] = transform[5];
    inverse[10] = transform[10];
    inverse[1] = transform[4];
    inverse[2] = transform[8];
    inverse[6] = transform[9];
    inverse[4] = transform[1];
    inverse[8] = transform[2];
    inverse[9] = transform[6];
    inverse[3] = -(inverse[0] * transform[3] + inverse[1] * transform[7] +
                   inverse[2] * transform[11]);
    inverse[7] = -(inverse[4] * transform[3] + inverse[5] * transform[7] +
                   inverse[6] * transform[11]);
    inverse[11] = -(inverse[8] * transform[3] + inverse[9] * transform[7] +
                    inverse[10] * transform[11]);
  } else {
    std::swap(inverse[1], inverse[4]);
    std::swap(inverse[2], inverse[8]);
    std::swap(inverse[6], inverse[9]);
    DType tmp[3]{};
    tmp[0] = -(inverse[0] * transform[3] + inverse[1] * transform[7] +
               inverse[2] * transform[11]);
    tmp[1] = -(inverse[4] * transform[3] + inverse[5] * transform[7] +
               inverse[6] * transform[11]);
    tmp[2] = -(inverse[8] * transform[3] + inverse[9] * transform[7] +
               inverse[10] * transform[11]);
    inverse[3] = tmp[0];
    inverse[7] = tmp[1];
    inverse[11] = tmp[2];
  }
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC DType clamp(DType const x, DType const lb,
                                                DType const ub) {
  return fmin(ub, fmax(lb, x));
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
set_constant_vec_3d(DType *const v, DType const constant) {
  v[0] = constant;
  v[1] = constant;
  v[2] = constant;
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
get_tri_interp_coeffs(DType const t0[3], DType coeffs[8]) {
  DType constexpr kOne{static_cast<DType>(1.0f)};
  DType const t1[3]{kOne - t0[0], kOne - t0[1], kOne - t0[2]};
  coeffs[0b000] = t1[0] * t1[1] * t1[2];
  coeffs[0b001] = t1[0] * t1[1] * t0[2];
  coeffs[0b010] = t1[0] * t0[1] * t1[2];
  coeffs[0b011] = t1[0] * t0[1] * t0[2];
  coeffs[0b100] = t0[0] * t1[1] * t1[2];
  coeffs[0b101] = t0[0] * t1[1] * t0[2];
  coeffs[0b110] = t0[0] * t0[1] * t1[2];
  coeffs[0b111] = t0[0] * t0[1] * t0[2];
}

template <typename DType>
INLINE_HOST_DEVICE_FUNC void
get_tri_interp_coeffs_grads(DType const t0[3], DType grad_x[8], DType grad_y[8],
                            DType grad_z[8]) {
  DType constexpr kOne{static_cast<DType>(1.0f)};
  DType const t1[3]{kOne - t0[0], kOne - t0[1], kOne - t0[2]};
#pragma once
  for (uint8_t i{0}; i < 8; ++i) {
    grad_x[i] = (i & 4 ? 1.0f : -1.0f) * (i & 2 ? t0[1] : t1[1]) *
                (i & 1 ? t0[2] : t1[2]);
    grad_y[i] = (i & 4 ? t0[0] : t1[0]) * (i & 2 ? 1.0f : -1.0f) *
                (i & 1 ? t0[2] : t1[2]);
    grad_z[i] = (i & 4 ? t0[0] : t1[0]) * (i & 2 ? t0[1] : t1[1]) *
                (i & 1 ? 1.0f : -1.0f);
  }
}

template <typename DType>
INLINE_DEVICE_FUNC void
find_point_on_ray_3d(DType const origin[3], DType const dir[3], DType const t,
                     DType p[3]) {
  utils::vec_scalar_mul_3d(dir, t, p);
  utils::vec_add_3d(origin, p, p);
}

} // namespace utils
} // namespace cuda