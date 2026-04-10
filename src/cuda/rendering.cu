#include "cuda/background.h"
#include "cuda/rendering.h"

namespace cuda {

DEVICE_FUNC math::Vector3f get_ray_color(
    Ray const &ray, DeviceHittableList const &world, int32_t const depth,
    utils::RandomNumberGenerator *const rng,
    SimpleSkyBackground const *const background) {
  if (depth <= 0) {
    return math::Vector3f::Zero();
  }

  float const inff{INFINITY};
  Ray scattered{ray};
  math::Vector3f attenuation{1.0f, 1.0f, 1.0f};
  int32_t d{depth};
  math::Vector3f curr_att{};
  for (; d > 0; --d) {
    HitRecordCuda hit_record{};
    if (world.Hit(scattered, Intervalf{0.001f, inff}, hit_record)) {
      if (nullptr != hit_record.material_property) {
        if (scatter(scattered, hit_record, rng, curr_att, scattered)) {
          attenuation = attenuation.CwiseProduct(curr_att);
        } else {
          return math::Vector3f::Zero();
        }
      }
    } else {
      break;
    }
  }

  return attenuation.CwiseProduct(background->GetRayColor(scattered));
}

__global__ void ray_tracing_kernal(Camera const camera,
                                   DeviceHittableList const world,
                                   SimpleSkyBackground const back_ground,
                                   uint64_t const image_width,
                                   uint64_t const image_height,
                                   RayTracerConfig const config,
                                   uint8_t *const rendered_image) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  if (u >= image_width || v >= image_height) {
    return;
  }

  uint64_t const pix_idx{u + v * image_width};
  uint8_t *const pixel{rendered_image + pix_idx * 3UL};
  utils::RandomNumberGenerator rendering_rng{pix_idx * 2654435761UL};
  utils::RandomNumberGenerator camera_rng{pix_idx * 3784435761UL};
  Intervalf intensity{0.000, 0.999};
  math::Vector3f color{0.0f, 0.0f, 0.0f};
  for (int32_t i{0}; i < config.samples_per_pixel; ++i) {
    // Ray const ray{camera.getDefocusPerturbedRay(
    //     PixCoord(u, v), config.max_sample_pert, &camera_rng)};
    math::Vector3f const dir{camera.getRotation() *
                           camera.getPerturbedPixelCameraCoordinate(
                               {u, v}, config.max_sample_pert, &camera_rng)};
    Ray const ray{camera.getPosition(), dir};
    color += get_ray_color(ray, world, config.max_depth, &rendering_rng,
                           &back_ground);
  }
  color /= config.samples_per_pixel;
  pixel[0] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.z(), 2.0f)), 0.0f, 255.0f));
  pixel[1] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.y(), 2.0f)), 0.0f, 255.0f));
  pixel[2] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.x(), 2.0f)), 0.0f, 255.0f));
}

__global__ void ray_tracing_using_smem_kernal(
    Camera const camera, DeviceHittableList const world,
    SimpleSkyBackground const back_ground, uint64_t const image_width,
    uint64_t const image_height, RayTracerConfig const config,
    uint8_t *const rendered_image) {
  extern __shared__ Sphere smem_spheres[];
  if (0U == threadIdx.x && 0U == threadIdx.y) {
    memcpy(smem_spheres, world.objects().d__sphere,
           sizeof(Sphere) * world.objects().num_spheres);
  }
  __syncthreads();

  DeviceHittableList shared_world{};
  shared_world.objects().d__sphere = smem_spheres;
  shared_world.objects().num_spheres = world.objects().num_spheres;

  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  if (u >= image_width || v >= image_height) {
    return;
  }

  uint64_t const pix_idx{u + v * image_width};
  uint8_t *const pixel{rendered_image + pix_idx * 3UL};
  utils::RandomNumberGenerator rendering_rng{pix_idx * 2654435761UL};
  utils::RandomNumberGenerator camera_rng{pix_idx * 3784435761UL};
  Intervalf intensity{0.000, 0.999};
  math::Vector3f color{0.0f, 0.0f, 0.0f};
  for (int32_t i{0}; i < config.samples_per_pixel; ++i) {
    // Ray const ray{camera.getDefocusPerturbedRay(
    //     PixCoord(u, v), config.max_sample_pert, &camera_rng)};
    math::Vector3f const dir{camera.getRotation() *
                           camera.getPerturbedPixelCameraCoordinate(
                               {u, v}, config.max_sample_pert, &camera_rng)};
    Ray const ray{camera.getPosition(), dir};
    color += get_ray_color(ray, shared_world, config.max_depth, &rendering_rng,
                           &back_ground);
  }
  color /= config.samples_per_pixel;
  pixel[0] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.z(), 2.0f)), 0.0f, 255.0f));
  pixel[1] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.y(), 2.0f)), 0.0f, 255.0f));
  pixel[2] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.x(), 2.0f)), 0.0f, 255.0f));
}

__global__ void ray_sampling_kernal(Camera const camera,
                                    uint64_t const image_width,
                                    uint64_t const image_height,
                                    RayTracerConfig const config,
                                    float *const sampled_ray_map) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  uint64_t const s{blockIdx.z * blockDim.z + threadIdx.z};
  if (u >= image_width || v >= image_height || s >= config.samples_per_pixel) {
    return;
  }

  // TODO Support defocus sampling
  uint64_t const ray_idx{s + v * config.samples_per_pixel +
                         u * config.samples_per_pixel * image_height};
  utils::RandomNumberGenerator rng{ray_idx * 3784435761UL};
  math::Vector3f const dir{camera.getRotation() *
                           camera.getPerturbedPixelCameraCoordinate(
                               {u, v}, config.max_sample_pert, &rng)};
  float *const ray{sampled_ray_map + ray_idx * 3UL};
  memcpy(ray, dir.data(), sizeof(float) * 3UL);
}

__global__ void hitting_kernal(
    Camera const camera, DeviceHittableList const world,
    SimpleSkyBackground const back_ground, uint64_t const image_width,
    uint64_t const image_height, RayTracerConfig const config,
    float const *const ray_map, float *const color_map) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  uint64_t const s{blockIdx.z * blockDim.z + threadIdx.z};
  if (u >= image_width || v >= image_height || s >= config.samples_per_pixel) {
    return;
  }

  uint64_t const ray_idx{s + v * config.samples_per_pixel +
                         u * config.samples_per_pixel * image_height};
  float const *const ray_dir{ray_map + ray_idx * 3UL};
  float *const color{color_map + ray_idx * 3UL};
  Ray const ray{camera.getPosition(),
                math::Vector3f{ray_dir[0], ray_dir[1], ray_dir[2]}};
  utils::RandomNumberGenerator rng{ray_idx * 3784435761UL};
  math::Vector3f const color_vec{
      get_ray_color(ray, world, config.max_depth, &rng, &back_ground)};
  memcpy(color, color_vec.data(), sizeof(float) * 3UL);
}

__global__ void color_blending_kernal(uint64_t const image_width,
                                      uint64_t const image_height,
                                      RayTracerConfig const config,
                                      float const *const color_map,
                                      uint8_t *const rendered_image) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  if (u >= image_width || v >= image_height) {
    return;
  }

  uint64_t const pix_idx{u + v * image_width};
  uint8_t *const pixel{rendered_image + pix_idx * 3UL};
  uint64_t const ray_idx{v * config.samples_per_pixel +
                         u * config.samples_per_pixel * image_height};
  float const *curr_color{color_map + ray_idx * 3UL};
  math::Vector3f color{0.0f, 0.0f, 0.0f};
  for (uint64_t i{0UL}; i < config.samples_per_pixel; ++i, curr_color += 3) {
    color.x() += curr_color[0];
    color.y() += curr_color[1];
    color.z() += curr_color[2];
  }
  color /= config.samples_per_pixel;
  Intervalf intensity{0.000, 0.999};
  pixel[0] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.z(), 2.0f)), 0.0f, 255.0f));
  pixel[1] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.y(), 2.0f)), 0.0f, 255.0f));
  pixel[2] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(color.x(), 2.0f)), 0.0f, 255.0f));
}

void render(Camera const &camera, DeviceHittableList const &world,
            RayTracerConfig const &config,
            SimpleSkyBackground const back_ground, uint64_t const image_width,
            uint64_t const image_height, uint8_t *const d__rendered_image) {
  dim3 const block_dim{BLOCK_DIM_X, BLOCK_DIM_Y};
  dim3 const grid_dim{GRID_DIM(image_width, block_dim.x),
                      GRID_DIM(image_height, block_dim.y)};
  // std::cout << "Start rendering " << image_width << "x" << image_height
  //           << std::endl;
  // ray_tracing_kernal<<<grid_dim, block_dim>>>(camera, world, back_ground,
  //                                             image_width, image_height,
  //                                             config, d__rendered_image);
  uint64_t const smem_size{sizeof(Sphere) * world.objects().num_spheres};
  ray_tracing_using_smem_kernal<<<grid_dim, block_dim, smem_size>>>(
      camera, world, back_ground, image_width, image_height, config,
      d__rendered_image);
  CUDA_CHECK(cudaDeviceSynchronize());
  // std::cout << "Rendering done. " << std::endl;
}

void render2(Camera const &camera, DeviceHittableList const &world,
             RayTracerConfig const &config,
             SimpleSkyBackground const back_ground, uint64_t const image_width,
             uint64_t const image_height, uint8_t *const d__rendered_image) {
  uint64_t const sample_buffer_size{image_width * image_height *
                                    config.samples_per_pixel * sizeof(float) *
                                    3UL};
  float *d__ray_map{};
  float *d__color_map{};
  CUDA_CHECK(cudaMalloc((void **)&d__ray_map, sample_buffer_size));
  CUDA_CHECK(cudaMalloc((void **)&d__color_map, sample_buffer_size));
  CUDA_CHECK(cudaDeviceSynchronize());

  dim3 const sampling_block_dim{8, 8, 16};
  dim3 const sampling_grid_dim{
      GRID_DIM(image_width, sampling_block_dim.x),
      GRID_DIM(image_height, sampling_block_dim.y),
      GRID_DIM(config.samples_per_pixel, sampling_block_dim.z)};
  dim3 const blending_block_dim{BLOCK_DIM_X, BLOCK_DIM_Y};
  dim3 const blending_grid_dim{GRID_DIM(image_width, blending_block_dim.x),
                               GRID_DIM(image_height, blending_block_dim.y)};
  // std::cout << "Start rendering " << image_width << "x" << image_height
  //           << std::endl;
  ray_sampling_kernal<<<sampling_grid_dim, sampling_block_dim>>>(
      camera, image_width, image_height, config, d__ray_map);
  CUDA_CHECK(cudaDeviceSynchronize());
  hitting_kernal<<<sampling_grid_dim, sampling_block_dim>>>(
      camera, world, back_ground, image_width, image_height, config, d__ray_map,
      d__color_map);
  CUDA_CHECK(cudaDeviceSynchronize());
  color_blending_kernal<<<blending_grid_dim, blending_block_dim>>>(
      image_width, image_height, config, d__color_map, d__rendered_image);
  CUDA_CHECK(cudaDeviceSynchronize());
  // std::cout << "Rendering done. " << std::endl;

  CUDA_CHECK(cudaFree(d__ray_map));
  CUDA_CHECK(cudaFree(d__color_map));
}

}  // namespace cuda