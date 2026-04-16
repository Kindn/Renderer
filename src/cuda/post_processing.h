#include "cuda/utils/common.h"
#include "cuda/utils/math.h"

namespace cuda {

class Bloom {
public:
  struct Config {
    uint64_t image_width{};
    uint64_t image_height{};
    int32_t kernal_size{5};
    float sigma{1.0f};
    float brightness_threshold{1.0f};
    float bloom_weight{0.3f};
    uint32_t num_mip_levels{6U};
  };

  explicit Bloom(Config const &config) : config_{config} { Initialize(); }

  void Initialize();

  void Apply(float const *const d__hdr_image,
             uint8_t *const d__output_image) noexcept;

  void Release();

private:
  void ExtractBrightRegion() noexcept;

  void BlurBrightRegion() noexcept;

  void ApplyBlurredBrightRegion(uint8_t *const d__output_image) noexcept;

  struct Data {
    float const *d__hdr_image{};
    float *d__gaussian_kernal{};
    std::vector<float *> d__mip_bright_region{};
    std::vector<float *> d__mip_blurred_bright_region{};
    std::vector<cudaArray_t> d__mip_blurred_bright_region_arr{};
    std::vector<cudaTextureObject_t> d__mip_blurred_bright_region_tex{};
    cudaTextureObject_t *d__tex_obj_list{};
  };

private:
  Config config_{};
  Data data_{};
};

} // namespace cuda