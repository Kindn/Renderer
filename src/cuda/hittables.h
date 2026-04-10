#pragma once

#include "cuda/materials.h"
#include "cuda/utils/common.h"

namespace cuda {

class Sphere {
 public:
  HOST_DEVICE_FUNC Sphere(MaterialType const &material_type,
                          void *const material_property) noexcept;

  HOST_DEVICE_FUNC Sphere(math::Vector3f const &center, float const radius,
                          MaterialType const &material_type,
                          void *const material_property) noexcept;

  HOST_DEVICE_FUNC bool Hit(Ray const &ray, Intervalf const &interval,
                            HitRecordCuda &hit_record) const noexcept;

  HOST_DEVICE_FUNC void SetMaterial(MaterialType const &material_type,
                                    void *const material_property) noexcept;

  math::Vector3f const &center() const noexcept { return center_; }

  float radius() const noexcept { return radius_; }

  MaterialType material_type() const noexcept { return material_type_; }

  void const *material_property() const noexcept { return material_property_; }

  friend class HostHittableList;

 private:
  math::Vector3f center_{};
  float radius_{0.2f};
  MaterialType material_type_{MaterialType::MATERIAL_LAMBERTIAN};
  void *material_property_{nullptr};
};

class HostHittableList {
 public:
  struct Objects {
    std::vector<Sphere> spheres{};
  };

  void Release();

  Objects const &objects() const noexcept { return objects_; }

  Objects &objects() noexcept { return objects_; }

 private:
  Objects objects_{};
};

class DeviceHittableList {
 public:
  struct Objects {
    uint64_t num_spheres{};
    Sphere *d__sphere{};
  };

  DeviceHittableList() = default;

  bool Upload(HostHittableList const &host);

  void Release();

  DEVICE_FUNC bool Hit(Ray const &ray, Intervalf const &interval,
                       HitRecordCuda &hit_record) const noexcept;

  INLINE_HOST_DEVICE_FUNC Objects const &objects() const { return objects_; }

  INLINE_HOST_DEVICE_FUNC Objects &objects() { return objects_; }

 private:
  struct MaterialInfo {
    MaterialType type{};
    void *prop{};

    MaterialInfo() = default;

    MaterialInfo(MaterialType const _type, void *const _prop)
        : type{_type}, prop{_prop} {}
  };

  //! Only used by host functions
  class MaterialInfoVec {
   public:
    HOST_DEVICE_FUNC MaterialInfoVec() {};

    MaterialInfoVec(std::vector<MaterialInfo> const &v) {
      if (!v.empty()) {
        uint64_t const buffer_size{v.size() * sizeof(MaterialInfo)};
        data_ = (MaterialInfo *)malloc(buffer_size);
        assert(nullptr != data_);
        memcpy(data_, v.data(), buffer_size);
        size_ = v.size();
      }
    }

    HOST_DEVICE_FUNC ~MaterialInfoVec() {
#if !defined(__CUDA_ARCH__)
      clear();
#endif
    }

    void clear() {
      if (nullptr != data_) {
        free(data_);
        data_ = nullptr;
      }
      size_ = 0UL;
    }

    MaterialInfo &at(uint64_t const idx) { return data_[idx]; }

    MaterialInfo const &at(uint64_t const idx) const { return data_[idx]; }

    uint64_t size() const { return size_; }

   private:
    uint64_t size_{0UL};
    MaterialInfo *data_{nullptr};
  };

  struct ObjectMemoryInfo {
    HOST_DEVICE_FUNC ~ObjectMemoryInfo(){};
    MaterialInfoVec spheres{};
  };

  Objects objects_{};
  ObjectMemoryInfo obj_mem_info_{};
};

}  // namespace cuda