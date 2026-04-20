#pragma once

#include <opencv2/opencv.hpp>

#include "cuda/background.h"
#include "cuda/camera.h"
#include "cuda/hittables.h"
#include "cuda/post_processing.h"
#include "cuda/utils/math.h"

namespace cuda {

struct RayTracerConfig {
  int32_t max_depth{50};
  int32_t samples_per_pixel{100};
  float max_sample_pert{0.5f};
};

void render(Camera const &camera, DeviceHittableList const &world,
            RayTracerConfig const &config,
            SimpleSkyBackground const back_ground, uint64_t const image_width,
            uint64_t const image_height, uint8_t *const d__rendered_image);

void render2(Camera const &camera, DeviceHittableList const &world,
             RayTracerConfig const &config,
             SimpleSkyBackground const back_ground, uint64_t const image_width,
             uint64_t const image_height, uint8_t *const d__rendered_image);

void render(Camera const &camera, DeviceSchwarzschildSpace const &world,
            RayTracerConfig const &config, HdriSkyBackground const back_ground,
            uint64_t const image_width, uint64_t const image_height,
            uint8_t *const d__rendered_image);

void render2(Camera const &camera, HostSchwarzschildSpace const &host_world,
             DeviceSchwarzschildSpace const &device_world,
             RayTracerConfig const &config, HdriSkyBackground const back_ground,
             uint64_t const image_width, uint64_t const image_height,
             std::shared_ptr<Bloom> const &bloom, float const time,
             uint8_t *const d__rendered_image);

}  // namespace cuda
