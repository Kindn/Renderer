#pragma once

#include "math/matrix.h"

namespace math {

template <typename DType, uint32_t Dim, typename FunctorType>
class Rk4OdeSolver {
 public:
  typedef math::Matrix<DType, Dim, 1U> State;

  HOST_DEVICE_FUNC Rk4OdeSolver(){};

  HOST_DEVICE_FUNC void Step(FunctorType const &f, State const &x0,
                             DType const t0, DType const step,
                             State &dx) const {
    DType constexpr kTwo{static_cast<DType>(2)};
    DType constexpr kSix{static_cast<DType>(6)};
    DType const half_step{static_cast<DType>(0.5f * step)};
    DType const t_mid{t0 + half_step};
    DType const t1{t0 + step};
    State const k1{f(t0, x0)};
    State const k2{f(t_mid, x0 + half_step * k1)};
    State const k3{f(t_mid, x0 + half_step * k2)};
    State const k4{f(t1, x0 + step * k3)};

    dx = step * (k1 + kTwo * k2 + kTwo * k3 + k4) / kSix;
  }

  HOST_DEVICE_FUNC void Solve(FunctorType const &f, DType const t0,
                              State const &x0, DType const step,
                              uint32_t const num_steps, State *const x_vec) {
    x_vec[0] = x0;
    DType t{t0};
    State dx{};
    for (uint32_t i{0U}; i < num_steps; ++i) {
      Step(f, x_vec[i], t, step, dx);
      x_vec[i + 1U] = x_vec[i] + dx;
      t += step;
    }
  }
};

template <typename DType, uint32_t Dim, typename FunctorType>
class EulerOdeSolver {
 public:
  typedef math::Matrix<DType, Dim, 1U> State;

  HOST_DEVICE_FUNC EulerOdeSolver(){};

  HOST_DEVICE_FUNC void Step(FunctorType const &f, State const &x0,
                             DType const t0, DType const step,
                             State &dx) const {
    dx = step * f(t0, x0);
  }

  HOST_DEVICE_FUNC void Solve(FunctorType const &f, DType const t0,
                              State const &x0, DType const step,
                              uint32_t const num_steps, State *const x_vec) {
    x_vec[0] = x0;
    DType t{t0};
    State dx{};
    for (uint32_t i{0U}; i < num_steps; ++i) {
      Step(f, x_vec[i], t, step, dx);
      x_vec[i + 1U] = x_vec[i] + dx;
      t += step;
    }
  }
};

template <typename SolverType, typename FunctorType, typename StateType,
          typename DType>
HOST_DEVICE_FUNC void SolveOde(SolverType &solver, FunctorType const &f,
                               DType const t0, StateType const &x0,
                               DType const step, uint32_t const num_steps,
                               StateType *const x_vec) {
  x_vec[0] = x0;
  DType t{t0};
  StateType dx{};
  for (uint32_t i{0U}; i < num_steps; ++i) {
    solver.Step(f, x_vec[i], t, step, dx);
    x_vec[i + 1U] = x_vec[i] + dx;
    t += step;
  }
}

}  // namespace math