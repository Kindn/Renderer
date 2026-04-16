#include "camera.h"
#include "cuda/background.h"
#include "cuda/utils/math.h"
#include "math/quaterion.h"

namespace cuda {
namespace bhlite {

void render(Camera const &camera, HdriSkyBackground const back_ground,
            cudaTextureObject_t acc_disk_tex,
            uint64_t const image_width, uint64_t const image_height,
            uint8_t *const d__rendered_image);

}
}  // namespace cude