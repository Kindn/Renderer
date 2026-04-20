#include "cuda/post_processing.h"

namespace cuda {

__global__ void bright_region_extraction_kernal(float const *const hdr_image,
                                                uint64_t const image_width,
                                                uint64_t const image_height,
                                                float const threshold,
                                                float *const bright_region) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  if (u >= image_width || v >= image_height) {
    return;
  }

  uint64_t const pix_idx{u + v * image_width};
  float const *const pixel_in{hdr_image + pix_idx * 3UL};
  float *const pixel_out{bright_region + pix_idx * 4UL};
  memset(pixel_out, 0, sizeof(float) * 4UL);
  if (0.2126f * pixel_in[0] + 0.7152f * pixel_in[1] + 0.0722f * pixel_in[2] >=
      threshold) {
    // memcpy(pixel_out, pixel_in, sizeof(float) * 3UL);
    pixel_out[0] = fmin(1.0f, pixel_in[0]);
    pixel_out[1] = fmin(1.0f, pixel_in[1]);
    pixel_out[2] = fmin(1.0f, pixel_in[2]);
    pixel_out[3] = 0.0f;
  }
}

__global__ void half_downsampling_kernal(float const *const input,
                                         uint64_t const image_width,
                                         uint64_t const image_height,
                                         float *const output) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  if (u >= image_width / 2 || v >= image_height / 2) {
    return;
  }

  uint64_t const pix_idx{u + v * image_width / 2};
  float *const pix_out{output + pix_idx * 4UL};
  uint64_t const input_pixel_idxs[4]{
      u * 2UL + image_width * v * 2UL, u * 2UL + 1UL + image_width * v * 2UL,
      u * 2UL + image_width * (v * 2UL + 1UL),
      u * 2UL + 1UL + image_width * (v * 2UL + 1UL)};
#pragma unroll
  for (uint8_t i{0}; i < 4; ++i) {
    pix_out[i] = (input[input_pixel_idxs[0] * 4UL + i] +
                  input[input_pixel_idxs[1] * 4L + i] +
                  input[input_pixel_idxs[2] * 4UL + i] +
                  input[input_pixel_idxs[3] * 4UL + i]) *
                 0.25f;
  }
}

__global__ void blur_kernal(float const *const input,
                            uint64_t const image_width,
                            uint64_t const image_height,
                            float const *const kernal,
                            int32_t const kernal_size, float *const output) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  if (u >= image_width || v >= image_height) {
    return;
  }

  uint64_t const pix_idx{u + v * image_width};
  float *const pix_out{output + pix_idx * 4UL};
  int32_t const half_window_size{kernal_size / 2};
  uint64_t curr_u{};
  uint64_t curr_v{};
  uint64_t curr_pix_idx{};
  float sum[4]{0.0f, 0.0f, 0.0f, 0.0f};
  for (int32_t offset_x{-half_window_size}; offset_x <= half_window_size;
       ++offset_x) {
    for (int32_t offset_y{-half_window_size}; offset_y <= half_window_size;
         ++offset_y) {
      curr_u = static_cast<uint32_t>(utils::clamp<int32_t>(
          static_cast<int32_t>(u) + offset_x, 0, image_width - 1));
      curr_v = static_cast<uint32_t>(utils::clamp<int32_t>(
          static_cast<int32_t>(v) + offset_y, 0, image_height - 1));
      curr_pix_idx = curr_u + image_width * curr_v;
      float const *const input_pix{input + curr_pix_idx * 4UL};
      float const weight{kernal[(offset_x + half_window_size) +
                                kernal_size * (offset_y + half_window_size)]};
      sum[0] += weight * input_pix[0];
      sum[1] += weight * input_pix[1];
      sum[2] += weight * input_pix[2];
      sum[3] += weight * input_pix[3];
    }
  }
  memcpy(pix_out, sum, sizeof(float) * 4UL);
}

__global__ void bloom_kernal(float const *const hdr_image,
                             cudaTextureObject_t *const blurred_bright_region,
                             uint32_t const num_mip_levels,
                             uint64_t const image_width,
                             uint64_t const image_height,
                             float const bloom_weight,
                             uint8_t *const output_image) {
  uint64_t const u{blockIdx.x * blockDim.x + threadIdx.x};
  uint64_t const v{blockIdx.y * blockDim.y + threadIdx.y};
  if (u >= image_width || v >= image_height) {
    return;
  }

  uint64_t const pix_idx{u + v * image_width};
  float const *const pix_hdr{hdr_image + pix_idx * 3UL};
  uint8_t *const pix_out{output_image + pix_idx * 3UL};
  float bloom_eff[3]{0.0f, 0.0f, 0.0f};
  float const un{(static_cast<float>(u) + 0.5f) / (image_width - 1UL)};
  float const vn{(static_cast<float>(v) + 0.5f) / (image_height - 1UL)};
  for (uint32_t i{0U}; i < num_mip_levels; ++i) {
    float4 const tex{tex2D<float4>(blurred_bright_region[i], un, vn)};
    bloom_eff[0] += tex.x;
    bloom_eff[1] += tex.y;
    bloom_eff[2] += tex.z;
  }

  Intervalf const intensity{0.000f, 0.999f};
  pix_out[0] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(
                   pix_hdr[2] + bloom_weight * bloom_eff[2], 2.0f)),
      0.0f, 255.0f));
  pix_out[1] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(
                   pix_hdr[1] + bloom_weight * bloom_eff[1], 2.0f)),
      0.0f, 255.0f));
  pix_out[2] = static_cast<uint8_t>(utils::clamp<float>(
      256.0f * intensity.clamp(linearToGamma(
                   pix_hdr[0] + bloom_weight * bloom_eff[0], 2.0f)),
      0.0f, 255.0f));
}

void get_device_gaussian_kernal(int32_t const kernal_size, float const sigma,
                                float *const d__kernal) {
  int32_t const half_kernal_size{kernal_size / 2};
  float const sigma2{sigma * sigma};
  float const inv_sigma2_2{0.5f / sigma2};
  float *const h__kernal{new float[kernal_size * kernal_size]};
  float sum{0.0f};
  for (int32_t u{-half_kernal_size}; u <= half_kernal_size; ++u) {
    for (int32_t v{-half_kernal_size}; v <= half_kernal_size; ++v) {
      float const weight{std::exp(-(u * u + v * v) * inv_sigma2_2)};
      h__kernal[(u + half_kernal_size) + (v + half_kernal_size) * kernal_size] =
          weight;
      sum += weight;
    }
  }
  for (int32_t i{0}; i < kernal_size; ++i) {
    // h__kernal[i] /= sum;
    h__kernal[i] /= (2.0f * M_PIf32 * sigma2);
  }

  CUDA_CHECK(cudaMemcpy(d__kernal, h__kernal,
                        sizeof(float) * kernal_size * kernal_size,
                        cudaMemcpyHostToDevice));

  delete[] h__kernal;
}

void Bloom::Apply(float const *const d__hdr_image,
                  uint8_t *const d__output_image) noexcept {
  data_.d__hdr_image = d__hdr_image;
  ExtractBrightRegion();
  BlurBrightRegion();
  ApplyBlurredBrightRegion(d__output_image);
}

void Bloom::Initialize() {
  data_.d__mip_bright_region.resize(config_.num_mip_levels);
  data_.d__mip_blurred_bright_region.resize(config_.num_mip_levels);
  data_.d__mip_blurred_bright_region_arr.resize(config_.num_mip_levels);
  data_.d__mip_blurred_bright_region_tex.resize(config_.num_mip_levels);

  uint64_t curr_width{config_.image_width};
  uint64_t curr_height{config_.image_height};
  for (uint32_t i{0U}; i < config_.num_mip_levels;
       ++i, curr_width >>= 1, curr_height >>= 1) {
    uint64_t const curr_buffer_size{sizeof(float) * curr_width * curr_height *
                                    4UL};
    CUDA_CHECK(
        cudaMalloc((void **)&data_.d__mip_bright_region[i], curr_buffer_size));
    CUDA_CHECK(cudaMalloc((void **)&data_.d__mip_blurred_bright_region[i],
                          curr_buffer_size));
    cudaChannelFormatDesc const ch_desc{cudaCreateChannelDesc<float4>()};
    CUDA_CHECK(cudaMallocArray(&data_.d__mip_blurred_bright_region_arr[i],
                               &ch_desc, curr_width, curr_height));
    cudaResourceDesc res_desc{};
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = data_.d__mip_blurred_bright_region_arr[i];
    cudaTextureDesc tex_desc{};
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    cudaCreateTextureObject(&data_.d__mip_blurred_bright_region_tex[i],
                            &res_desc, &tex_desc, NULL);
  }

  CUDA_CHECK(cudaMalloc((void **)&data_.d__tex_obj_list,
                        sizeof(cudaTextureObject_t) * config_.num_mip_levels));
  CUDA_CHECK(cudaMemcpy(data_.d__tex_obj_list,
                        data_.d__mip_blurred_bright_region_tex.data(),
                        sizeof(cudaTextureObject_t) * config_.num_mip_levels,
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(
      cudaMalloc((void **)&data_.d__gaussian_kernal,
                 sizeof(float) * config_.kernal_size * config_.kernal_size));
  get_device_gaussian_kernal(config_.kernal_size, config_.sigma,
                             data_.d__gaussian_kernal);
}

void Bloom::Release() {
  for (auto &d__ptr : data_.d__mip_bright_region) {
    if (nullptr != d__ptr) {
      CUDA_CHECK(cudaFree(d__ptr));
      d__ptr = nullptr;
    }
  }
  for (auto &d__ptr : data_.d__mip_blurred_bright_region) {
    if (nullptr != d__ptr) {
      CUDA_CHECK(cudaFree(d__ptr));
      d__ptr = nullptr;
    }
  }
  for (auto &tex : data_.d__mip_blurred_bright_region_tex) {
    CUDA_CHECK(cudaDestroyTextureObject(tex));
  }
  for (auto &d__arr : data_.d__mip_blurred_bright_region_arr) {
    CUDA_CHECK(cudaFreeArray(d__arr));
  }

  if (nullptr != data_.d__gaussian_kernal) {
    CUDA_CHECK(cudaFree(data_.d__gaussian_kernal));
    data_.d__gaussian_kernal = nullptr;
  }

  if (nullptr != data_.d__tex_obj_list) {
    CUDA_CHECK(cudaFree(data_.d__tex_obj_list));
    data_.d__tex_obj_list = nullptr;
  }
}

void Bloom::ExtractBrightRegion() noexcept {
  if (0 == config_.num_mip_levels) {
    return;
  }

  dim3 const block_dim{BLOCK_DIM_X, BLOCK_DIM_Y};
  dim3 const grid_dim{GRID_DIM(config_.image_width, block_dim.x),
                      GRID_DIM(config_.image_height, block_dim.y)};
  bright_region_extraction_kernal<<<grid_dim, block_dim>>>(
      data_.d__hdr_image, config_.image_width, config_.image_height,
      config_.brightness_threshold, data_.d__mip_bright_region[0]);
  //* Construct mipmap
  uint64_t curr_width{config_.image_width};
  uint64_t curr_height{config_.image_height};
  for (uint32_t i{1U}; i < config_.num_mip_levels;
       ++i, curr_width >>= 1, curr_height >>= 1) {
    dim3 const curr_grid_dim{GRID_DIM(curr_width / 2, block_dim.x),
                             GRID_DIM(curr_height / 2, block_dim.y)};
    half_downsampling_kernal<<<curr_grid_dim, block_dim>>>(
        data_.d__mip_bright_region[i - 1U], curr_width, curr_height,
        data_.d__mip_bright_region[i]);
  }
}

void Bloom::BlurBrightRegion() noexcept {
  dim3 const block_dim{BLOCK_DIM_X, BLOCK_DIM_Y};
  uint64_t curr_width{config_.image_width};
  uint64_t curr_height{config_.image_height};
  for (uint32_t i{0U}; i < config_.num_mip_levels;
       ++i, curr_width >>= 1, curr_height >>= 1) {
    dim3 const curr_grid_dim{GRID_DIM(curr_width, block_dim.x),
                             GRID_DIM(curr_height, block_dim.y)};
    blur_kernal<<<curr_grid_dim, block_dim>>>(
        data_.d__mip_bright_region[i], curr_width, curr_height,
        data_.d__gaussian_kernal, config_.kernal_size,
        data_.d__mip_blurred_bright_region[i]);
    CUDA_CHECK(cudaMemcpy2DToArray(
        data_.d__mip_blurred_bright_region_arr[i], 0UL, 0UL,
        data_.d__mip_blurred_bright_region[i], curr_width * sizeof(float4),
        curr_width * sizeof(float4), curr_height, cudaMemcpyDeviceToDevice));
  }
}

void Bloom::ApplyBlurredBrightRegion(uint8_t *const d__output_image) noexcept {
  if (0 == config_.num_mip_levels) {
    return;
  }

  dim3 const block_dim{BLOCK_DIM_X, BLOCK_DIM_Y};
  dim3 const grid_dim{GRID_DIM(config_.image_width, block_dim.x),
                      GRID_DIM(config_.image_height, block_dim.y)};
  bloom_kernal<<<grid_dim, block_dim>>>(
      data_.d__hdr_image, data_.d__tex_obj_list, config_.num_mip_levels,
      config_.image_width, config_.image_height, config_.bloom_weight,
      d__output_image);
}

}  // namespace cuda