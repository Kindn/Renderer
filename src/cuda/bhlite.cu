#include "cuda/bhlite.h"
#include "cuda/hittables.h"

namespace cuda {
namespace bhlite {

INLINE_DEVICE_FUNC math::Vector3f get_accel(float const h2,
                                            math::Vector3f const r) {
  return -1.5f * h2 * powf(r.SquaredNorm(), -2.5f) * r;
}

INLINE_DEVICE_FUNC void apply_accretion_disk_color(
    math::Vector3f const &pos, math::Vector3f const &dir,
    cudaTextureObject_t acc_disk_tex, math::Vector3f &color) {
  float constexpr kInnerR{2.6f};
  float constexpr kOuterR{10.0f};
  float constexpr kWidth{kOuterR - kInnerR};
  float constexpr kThickness{0.2f};
  float constexpr kHalfThickness{0.5f * kThickness};
  math::Vector3f constexpr kColor{255.0f / 255.0f, 165.0f / 255.0f,
                                  153.0f / 255.0f};

  float const r_xy{hypot(pos.x(), pos.y())};
  if (r_xy > kOuterR || r_xy < kInnerR || fabs(pos.z()) > kHalfThickness) {
    return;
  }

  // float const density{(1.0f - (r_xy - kInnerR) / kWidth) *
  //                     (1.0f - fabs(pos.z()) / kHalfThickness)};
  // float const density{(1.0f - (r_xy - kInnerR) / kWidth)};
  // float const mapped_r{0.25f * (kOuterR - 2.0f * kInnerR + r_xy) /
  //                      (kOuterR - kInnerR)};
  // float const theta{atan2f(pos.y(), pos.x())};
  // float const density{tex2D<float>(acc_disk_tex, mapped_r * cosf(theta) +
  // 0.5f,
  //                                  mapped_r * sinf(theta) + 0.5f) *
  //                     fmax(0.0f, 1.0f - fabs(pos.z()) / kThickness)};
  float const theta{atan2f(pos.y(), pos.x())};
  float const u{theta / (2.0f * M_PIf32) + 0.5f};
  float const v{(r_xy - kInnerR) / kWidth};
  float const density{
      tex2D<float>(acc_disk_tex, u, v) * (1.0f - v) *
      fmax(0.0f, 1.0f - fabs(pos.z()) / (kThickness * powf((1.0f - v), 2.0f)))};

  color += density * kColor;
}

INLINE_DEVICE_FUNC bool get_ray_color(Ray const &ray,
                                      HdriSkyBackground const back_ground,
                                      cudaTextureObject_t acc_disk_tex,
                                      utils::RandomNumberGenerator *const rng,
                                      math::Vector3f &color) {
  float constexpr kStep{0.1f};
  float constexpr kMinStep{0.0001f};
  uint32_t constexpr kMaxNumSteps{1000U};

  math::Vector3f curr_pos{ray.getOrigin()};
  math::Vector3f curr_dir{ray.getDirection()};
  float const h2{curr_pos.Cross(curr_dir).SquaredNorm()};
  color.SetZero();
  for (uint32_t i{0U}; i < kMaxNumSteps; ++i) {
    math::Vector3f const acc{get_accel(h2, curr_pos)};
    float step{kStep};
    if (0U == i) {
      step *= rng->uniformReal(1.0e-6f, 1.0f);
    }
    step = utils::clamp(step, kMinStep, kStep);
    curr_dir += acc * step;
    curr_pos += curr_dir * step;

    if (curr_pos.SquaredNorm() <= 1.0f) {
      return true;
    }
    apply_accretion_disk_color(curr_pos, curr_dir, acc_disk_tex, color);
  }

  color += back_ground.GetRayColor(Ray(curr_pos, curr_dir));

  return false;
}

__global__ void rendering_kernal(Camera const camera,
                                 HdriSkyBackground const back_ground,
                                 cudaTextureObject_t acc_disk_tex,
                                 uint64_t const image_width,
                                 uint64_t const image_height,
                                 uint8_t *const rendered_image) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  if (u >= image_width || v >= image_height) {
    return;
  }

  uint64_t const pix_idx{u + v * image_width};
  uint8_t *const pixel{rendered_image + pix_idx * 3UL};
  math::Vector3f color{0.0f, 0.0f, 0.0f};
  utils::RandomNumberGenerator rng{pix_idx * 3784435761UL};
  uint32_t constexpr kNumSamples{1U};
  for (uint32_t i{0U}; i < kNumSamples; ++i) {
    Ray const ray{camera.getDefocusPerturbedRay(PixCoord(u, v), 0.0f, &rng)};
    math::Vector3f curr_color{};
    bool const hit_event_horizon{
        get_ray_color(ray, back_ground, acc_disk_tex, &rng, curr_color)};
    color += curr_color;
  }
  color /= kNumSamples;

  Intervalf intensity{0.000f, 0.999f};
  pixel[0] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.z(), 2.0f)), 0.0f, 255.0f));
  pixel[1] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.y(), 2.0f)), 0.0f, 255.0f));
  pixel[2] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.x(), 2.0f)), 0.0f, 255.0f));
}

void render(Camera const &camera, HdriSkyBackground const back_ground,
            cudaTextureObject_t acc_disk_tex, uint64_t const image_width,
            uint64_t const image_height, uint8_t *const d__rendered_image) {
  dim3 const block_dim{BLOCK_DIM_X, BLOCK_DIM_Y};
  dim3 const grid_dim{GRID_DIM(image_width, block_dim.x),
                      GRID_DIM(image_height, block_dim.y)};
  rendering_kernal<<<grid_dim, block_dim>>>(camera, back_ground, acc_disk_tex,
                                            image_width, image_height,
                                            d__rendered_image);
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace bhlite
}  // namespace cuda