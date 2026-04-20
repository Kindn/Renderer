#pragma once

#include "cuda/materials.h"
#include "cuda/utils/common.h"
#include "cuda/utils/math.h"
#include "geometry/polyhedrons.h"
#include "math/ode_solvers.h"
#include "math/math.h"

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

class BlackHole {
 public:
  enum AccretionDiskTextureType { TRANSMISSION_PROBABILITY, PARTICLE_DENSITY };

  struct HostAccretionDiskTexture {
    float *h__texture{};
    uint32_t rows{};
    uint32_t cols{};
    AccretionDiskTextureType type{
        AccretionDiskTextureType::TRANSMISSION_PROBABILITY};
  };

  struct Config {
    /* Position of the singularity */
    math::Vector3f position{};
    /* Rotation of the accretion disk */
    math::Quaternionf rotation{};
    /* The Schwarszchild radius */
    float sr{1.0f};
    /* Inner radius of the accretion disk */
    float inner_disk_radius{1.2f};
    /* Inner radius of the accretion disk */
    float outer_disk_radius{2.5f};
    /* Attentuation of the accretion disk */
    math::Vector3f disk_attenuation{};
    /* Emission of the accretion disk */
    math::Vector3f disk_emission{};
    /* Thickness of the accretion disk */
    float disk_thickness{0.01f};
    /* The maximum range of effect of consideration */
    float max_eff_range{100.0f};
    /* Scaling factor of brightness relative to particle density. Only enabled
     * when texture type is PARTICLE_DENSITY */
    float brightness_scale{2.0f};

    INLINE_HOST_DEVICE_FUNC bool Check() const {
      return sr > 0.0f && inner_disk_radius > 0.0f &&
             outer_disk_radius > 0.0f && outer_disk_radius > inner_disk_radius;
    }
  };

  explicit BlackHole(
      Config const &config,
      HostAccretionDiskTexture const &host_acc_disk_tex) noexcept;

  DEVICE_FUNC bool Hit(Ray const &ray, Intervalf const &interval,
                       float const time,
                       HitRecordCuda &hit_record) const noexcept;

  HOST_DEVICE_FUNC Config const &config() const { return config_; }

  void Release();

  INLINE_HOST_DEVICE_FUNC float GetDistToSphereBoundary(
      math::Vector3f const &origin) const noexcept {
    return (origin - config_.position).Norm() - config_.outer_disk_radius;
  }

 private:
  DEVICE_FUNC bool HitEventHorizon(Ray const &ray,
                                   Intervalf const &interval) const;

  DEVICE_FUNC bool HitEventHorizon(Ray const &ray, Intervalf const &interval,
                                   float &t, math::Vector3f &p,
                                   math::Vector3f &outer_normal) const;

  DEVICE_FUNC bool HitAccretionDisk(Ray const &ray, Intervalf const &interval,
                                    float const time, float &t,
                                    math::Vector3f &p,
                                    math::Vector3f &outer_normal,
                                    float &tex_value, float &r_xy,
                                    float &redshift) const;

  void UploadTexture(HostAccretionDiskTexture const host_acc_disk_tex);

 private:
  struct DeviceAccretionDiskTexture {
    cudaArray_t d__tex_arr{};
    cudaTextureObject_t d__tex{};
    uint32_t rows{};
    uint32_t cols{};
    AccretionDiskTextureType type{
        AccretionDiskTextureType::TRANSMISSION_PROBABILITY};
  };

  Config config_{};
  math::Vector3f disk_normal_{};
  math::Vector3f origin_top_surface_{};
  math::Vector3f origin_bottom_surface_{};
  float odr2_{};
  float idr2_{};
  float half_disk_thickness_{};
  DeviceAccretionDiskTexture device_acc_disk_tex_{};
  float disk_width_{};
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
    HOST_DEVICE_FUNC MaterialInfoVec(){};

    MaterialInfoVec(std::vector<MaterialInfo> const &v) {
      if (!v.empty()) {
        uint64_t const buffer_size{v.size() * sizeof(MaterialInfo)};
        data_ = (MaterialInfo *)malloc(buffer_size);
        // assert(nullptr != data_);
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

class SchwarzschildGeodesicFunctor {
 public:
  typedef math::Vector6f State;
  struct Config {
    float sr{1.0f};
    math::Vector3f position{0.0f, 0.0f, 0.0f};

    INLINE_HOST_DEVICE_FUNC bool Check() const { return sr > 0.0f; }
  };

  HOST_DEVICE_FUNC explicit SchwarzschildGeodesicFunctor(
      Config const &config) noexcept;

  HOST_DEVICE_FUNC State operator()(float const t,
                                    State const &x0) const noexcept;

  HOST_DEVICE_FUNC void SetAngularMomentum(Ray const &ray) noexcept;

  HOST_DEVICE_FUNC void SetAngularMomentum(float const am) noexcept {
    am_ = am;
    am2_ = am_ * am_;
  }

 private:
  Config config_{};
  float am_{};
  float am2_{};
};

struct SchwarzschildSpaceCameraPrior {
  math::Vector3f position{};
  math::Quaternionf rotation{};
  float nearest_obj_boundary_dist{0.0f};
};

class HostSchwarzschildSpace {
 public:
  struct Objects {
    std::vector<BlackHole> black_holes{};
  };

  HostSchwarzschildSpace() = default;

  void Release();

  Objects const &objects() const noexcept { return objects_; }

  Objects &objects() noexcept { return objects_; }

  SchwarzschildSpaceCameraPrior GetCameraPrior(
      math::Vector3f const &camera_position,
      math::Quaternionf const &camera_rotation) const noexcept;

 private:
  Objects objects_{};
};

class DeviceSchwarzschildSpace {
 public:
  struct Config {
    float max_marching_time{100.0f};

    HOST_DEVICE_FUNC bool Check() const { return max_marching_time >= 0.0f; }
  };

  struct Objects {
    uint64_t num_black_holes{};
    BlackHole *d__black_holes{};
    SchwarzschildGeodesicFunctor *d__geodesics{};
  };

  DeviceSchwarzschildSpace(Config const &config) noexcept : config_{config} {
    // assert(config_.Check());
  }

  bool Upload(HostSchwarzschildSpace const &host);

  void Release();

  DEVICE_FUNC bool Hit(Ray const &ray, Intervalf const &interval,
                       HitRecordCuda &hit_record) const noexcept;

  DEVICE_FUNC bool Hit(Ray const &ray, Intervalf const &interval,
                       float const *const __restrict__ r_list,
                       float *const smem_h2_map,
                       float const time,
                       HitRecordCuda &hit_record) const noexcept;

  INLINE_HOST_DEVICE_FUNC Objects const &objects() const { return objects_; }

  INLINE_HOST_DEVICE_FUNC Objects &objects() { return objects_; }

  INLINE_HOST_DEVICE_FUNC void SetCameraPrior(
      SchwarzschildSpaceCameraPrior const &camera_prior) noexcept {
    camera_prior_ = camera_prior;
  }

 private:
  DEVICE_FUNC bool HitBlackHoles(Ray const &ray, Intervalf const &interval,
                                 float const time,
                                 HitRecordCuda &hit_record) const noexcept;

  DEVICE_FUNC math::Vector3f GetAcc(float const *const h2_list,
                                    math::Vector3f const &pos) const noexcept;

  DEVICE_FUNC float StepAdaptiveEuler(float const *const h2_list,
                                      math::Vector3f &pos,
                                      math::Vector3f &dir) const noexcept;

 private:
  Config config_{};
  Objects objects_{};
  SchwarzschildSpaceCameraPrior camera_prior_{};
};

}  // namespace cuda