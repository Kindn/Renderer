#include "cuda/hittables.h"

namespace cuda {

HOST_DEVICE_FUNC Sphere::Sphere(MaterialType const &material_type,
                                void *const material_property) noexcept {
  SetMaterial(material_type, material_property);
}

HOST_DEVICE_FUNC Sphere::Sphere(math::Vector3f const &center,
                                float const radius,
                                MaterialType const &material_type,
                                void *const material_property) noexcept
    : center_{center}, radius_{radius} {
  SetMaterial(material_type, material_property);
}

HOST_DEVICE_FUNC bool Sphere::Hit(Ray const &ray, Intervalf const &interval,
                                  HitRecordCuda &hit_record) const noexcept {
  const math::Vector3f &ray_dir = ray.getDirection();
  const math::Vector3f &ray_ori = ray.getOrigin();
  const math::Vector3f oc = center_ - ray_ori;
  const float a = ray_dir.SquaredNorm();
  const float c = oc.SquaredNorm() - radius_ * radius_;
  const float h = oc.Dot(ray_dir);

  const float delta = h * h - a * c;
  if (delta < 0) {
    return false;
  }

  // Find the nearest root that lies in the acceptable range
  const float sqrtd = sqrtf(delta);
  float root = (h - sqrtd) / a;
  if (!interval.surrounds(root)) {
    root = (h + sqrtd) / a;
    if (!interval.surrounds(root)) {
      return false;
    }
  }
  hit_record.t = root;
  hit_record.p = ray.at(root);
  const math::Vector3f outward_normal = (hit_record.p - center_).Normalized();
  hit_record.SetFaceNormal(&ray, outward_normal);
  hit_record.material = material_type_;
  hit_record.material_property = material_property_;

  return true;
}

HOST_DEVICE_FUNC void Sphere::SetMaterial(
    MaterialType const &material_type, void *const material_property) noexcept {
  assert(nullptr != material_property);
  material_type_ = material_type;
  material_property_ = material_property;
}

void HostHittableList::Release() {
  for (auto &m : objects_.spheres) {
    if (nullptr != m.material_property_) {
      if (MaterialType::MATERIAL_DIELECTRIC == m.material_type()) {
        delete reinterpret_cast<DielectricProperty const *>(
            m.material_property_);
      } else if (MaterialType::MATERIAL_LAMBERTIAN == m.material_type()) {
        delete reinterpret_cast<LambertianProperty const *>(
            m.material_property_);
      } else if (MaterialType::MATERIAL_METAL == m.material_type()) {
        delete reinterpret_cast<MetalProperty const *>(m.material_property_);
      }

      m.material_property_ = nullptr;
    }
  }
  objects_.spheres.clear();
}

bool DeviceHittableList::Upload(HostHittableList const &host) {
  auto const &objects{host.objects()};

  //* Upload spheres
  {
    uint64_t const num_objs{objects.spheres.size()};
    uint64_t const buffer_size{sizeof(Sphere) * num_objs};

    objects_.num_spheres = num_objs;
    Sphere *const host_to_device_buffer{(Sphere *)malloc(buffer_size)};
    if (nullptr == host_to_device_buffer) {
      return false;
    }
    memcpy(host_to_device_buffer, objects.spheres.data(), buffer_size);

    CUDA_CHECK(cudaMalloc((void **)&objects_.d__sphere, buffer_size));
    std::vector<MaterialInfo> sphere_material_info{};
    sphere_material_info.reserve(num_objs);
    for (uint64_t i{0UL}; i < num_objs; ++i) {
      auto const &obj{objects.spheres.at(i)};
      void *d__material_prop_ptr{};
      uint64_t prop_buffer_size{};
      if (MaterialType::MATERIAL_DIELECTRIC == obj.material_type()) {
        prop_buffer_size = sizeof(DielectricProperty);
      } else if (MaterialType::MATERIAL_LAMBERTIAN == obj.material_type()) {
        prop_buffer_size = sizeof(LambertianProperty);
      } else if (MaterialType::MATERIAL_METAL == obj.material_type()) {
        prop_buffer_size = sizeof(MetalProperty);
      } else {
        return false;
      }
      CUDA_CHECK(cudaMalloc((void **)&d__material_prop_ptr, prop_buffer_size));
      CUDA_CHECK(cudaMemcpy(d__material_prop_ptr, obj.material_property(),
                            prop_buffer_size, cudaMemcpyHostToDevice));
      host_to_device_buffer[i].SetMaterial(obj.material_type(),
                                           d__material_prop_ptr);
      sphere_material_info.emplace_back(obj.material_type(),
                                        d__material_prop_ptr);
    }
    CUDA_CHECK(cudaMemcpy(objects_.d__sphere, host_to_device_buffer,
                          buffer_size, cudaMemcpyHostToDevice));

    free(host_to_device_buffer);
  }

  return true;
}

void DeviceHittableList::Release() {
  if (nullptr != objects_.d__sphere) {
    CUDA_CHECK(cudaFree(objects_.d__sphere));
    objects_.d__sphere = nullptr;
  }

  for (uint64_t i{0UL}; i < obj_mem_info_.spheres.size(); ++i) {
    auto &m{obj_mem_info_.spheres.at(i)};
    if (nullptr != m.prop) {
      CUDA_CHECK(cudaFree(m.prop));
      m.prop = nullptr;
    }
  }
  obj_mem_info_.spheres.clear();
  objects_.num_spheres = 0UL;
}

DEVICE_FUNC bool DeviceHittableList::Hit(
    Ray const &ray, Intervalf const &interval,
    HitRecordCuda &hit_record) const noexcept {
  HitRecordCuda temp_rec;
  bool hit_anything = false;
  float closest_so_far = interval.max();

  //* Hit spheres
  for (uint64_t i{0UL}; i < objects_.num_spheres; ++i) {
    auto const object{objects_.d__sphere + i};
    if (object->Hit(ray, Intervalf(interval.min(), closest_so_far), temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      hit_record = temp_rec;
    }
  }

  return hit_anything;
}

}  // namespace cuda