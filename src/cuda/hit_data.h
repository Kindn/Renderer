#pragma once

#include "utils.h"
#include "Ray.h"

namespace cuda {

enum MaterialType {
  MATERIAL_DIELECTRIC = 0,
  MATERIAL_LAMBERTIAN,
  MATERIAL_METAL
};

struct HitRecordCuda {
  math::Vector3f p{};
  math::Vector3f normal{};
  MaterialType material{};
  void *material_property{nullptr};
  float t{};
  bool front_face{};

  HOST_DEVICE_FUNC void SetFaceNormal(const Ray *const ray,
                                      const math::Vector3f &outward_normal) {
    front_face = (ray->getDirection().Dot(outward_normal) < 0);
    normal = front_face ? outward_normal : -outward_normal;
  }
};

}