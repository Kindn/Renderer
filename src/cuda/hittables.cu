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
  // assert(nullptr != material_property);
  material_type_ = material_type;
  material_property_ = material_property;
}

BlackHole::BlackHole(Config const &config,
                     HostAccretionDiskTexture const &host_acc_disk_tex) noexcept
    : config_{config} {
  // assert(config_.Check());
  disk_normal_ = config_.rotation * math::Vector3f::UnitZ();
  odr2_ = config_.outer_disk_radius * config_.outer_disk_radius;
  idr2_ = config_.inner_disk_radius * config_.inner_disk_radius;
  half_disk_thickness_ = 0.5f * config_.disk_thickness;
  origin_top_surface_ = config_.position + half_disk_thickness_ * disk_normal_;
  origin_bottom_surface_ =
      config_.position - half_disk_thickness_ * disk_normal_;
  UploadTexture(host_acc_disk_tex);
}

DEVICE_FUNC bool BlackHole::Hit(Ray const &ray, Intervalf const &interval,
                                HitRecordCuda &hit_record) const noexcept {
  // float t_h{0.0f};
  // math::Vector3f n_h{};
  // math::Vector3f p_h{};
  bool const hit_h{HitEventHorizon(ray, interval)};
  float t_d{0.0f};
  math::Vector3f n_d{};
  math::Vector3f p_d{};
  float tex_value{};
  float r_xy{};
  bool const hit_d{
      HitAccretionDisk(ray, interval, t_d, p_d, n_d, tex_value, r_xy)};
  bool const hit_anything{hit_h || hit_d};

  if (hit_h && !hit_d) {
    // hit_record.SetFaceNormal(&ray, n_h);
    // hit_record.t = t_h;
    // hit_record.p = p_h;
    hit_record.color.SetZero();
    hit_record.emitted.SetZero();
    hit_record.scattered = false;
    hit_record.final_dir = ray.getDirection();
    hit_record.is_event_horizon = true;
  } else if (!hit_h && hit_d) {
    hit_record.SetFaceNormal(&ray, n_d);
    hit_record.t = t_d;
    hit_record.p = p_d;
    float const scale{(r_xy - config_.outer_disk_radius) /
                      (config_.inner_disk_radius - config_.outer_disk_radius)};
    float const density{tex_value * scale * scale};
    hit_record.color.x() =
        fmin(1.0f, scale * config_.disk_attenuation.x() + 1.0f - scale);
    hit_record.color.y() =
        fmin(1.0f, scale * config_.disk_attenuation.y() + 1.0f - scale);
    hit_record.color.z() =
        fmin(1.0f, scale * config_.disk_attenuation.z() + 1.0f - scale);
    hit_record.emitted =
        AccretionDiskTextureType::PARTICLE_DENSITY == device_acc_disk_tex_.type
            ? config_.disk_emission * density * 2.0f * powf(r_xy, -3.0f) *
                  config_.brightness_scale
            : config_.disk_emission;
    hit_record.scattered = (AccretionDiskTextureType::PARTICLE_DENSITY ==
                            device_acc_disk_tex_.type);
    hit_record.final_dir = ray.getDirection();
    hit_record.is_event_horizon = false;
  } else if (hit_h && hit_d) {  //! Impossible outside the event horizon
    // if (t_h < t_d) {
    //   hit_record.SetFaceNormal(&ray, n_h);
    //   hit_record.t = t_h;
    //   hit_record.p = p_h;
    //   hit_record.color.SetZero();
    //   hit_record.emitted.SetZero();
    //   hit_record.scattered = false;
    //   hit_record.final_dir = ray.getDirection();
    // } else {
    //   hit_record.SetFaceNormal(&ray, n_d);
    //   hit_record.t = t_d;
    //   hit_record.p = p_d;
    //   hit_record.color.SetConstant(1.0f - tex_value);
    //   hit_record.emitted =
    //       AccretionDiskTextureType::PARTICLE_DENSITY ==
    //               device_acc_disk_tex_.type
    //           ? config_.disk_color * tex_value * config_.brightness_scale
    //           : config_.disk_color;
    //   hit_record.scattered = (AccretionDiskTextureType::PARTICLE_DENSITY ==
    //                           device_acc_disk_tex_.type);
    //   hit_record.final_dir = ray.getDirection();
    // }
  }

  return hit_anything;
}

void BlackHole::Release() {
  CUDA_CHECK(cudaDestroyTextureObject(device_acc_disk_tex_.d__tex));
  CUDA_CHECK(cudaFreeArray(device_acc_disk_tex_.d__tex_arr));
}

DEVICE_FUNC bool BlackHole::HitEventHorizon(Ray const &ray,
                                            Intervalf const &interval) const {
  float const dist2{(ray.at(interval.max()) - config_.position).SquaredNorm()};
  if (dist2 <= config_.sr * config_.sr) {
    return true;
  }

  return false;
}

DEVICE_FUNC bool BlackHole::HitEventHorizon(
    Ray const &ray, Intervalf const &interval, float &t, math::Vector3f &p,
    math::Vector3f &outer_normal) const {
  float const dist{(ray.at(interval.min()) - config_.position).Norm()};
  if (dist > config_.sr + interval.size()) {
    return false;
  } else if (dist - config_.sr < 1.0e-6 * config_.sr) {
    t = 0.0f;
    p = ray.getOrigin();
    outer_normal = (p - config_.position).Normalized();
    return true;
  }

  const math::Vector3f &ray_dir = ray.getDirection();
  const math::Vector3f &ray_ori = ray.getOrigin();
  const math::Vector3f oc = config_.position - ray_ori;
  const float a = ray_dir.SquaredNorm();
  const float c = oc.SquaredNorm() - config_.sr * config_.sr;
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

  t = root;
  p = ray.at(root);
  outer_normal = (p - config_.position).Normalized();

  return true;
}

DEVICE_FUNC bool BlackHole::HitAccretionDisk(
    Ray const &ray, Intervalf const &interval, float &t, math::Vector3f &p,
    math::Vector3f &outer_normal, float &tex_value, float &r_xy) const {
  float const d_dot_n{ray.getDirection().Dot(disk_normal_)};
  math::Vector3f const op{(config_.position - ray.getOrigin())};
  float const op_dot_n{op.Dot(disk_normal_)};
  if (fabs(d_dot_n) <= 1.0e-16) {
    return false;
  }

  float const root{op_dot_n / d_dot_n};
  if (!interval.surrounds(root)) {
    return false;
  }

  math::Vector3f const ipt{ray.at(root)};
  float const dist2{(ipt - config_.position).SquaredNorm()};
  if (dist2 > odr2_ || dist2 < idr2_) {
    return false;
  }

  t = root;
  p = ray.at(t);
  math::Vector3f const p_loc{config_.rotation.Conjugated() *
                             (p - config_.position)};
  float const theta{atan2f(p_loc.y(), p_loc.x())};
  r_xy = hypotf(p_loc.x(), p_loc.y());
  // float const mapped_r{
  //     0.25f *
  //     (config_.outer_disk_radius - 2.0f * config_.inner_disk_radius + r_xy) /
  //     (config_.outer_disk_radius - config_.inner_disk_radius)};
  // tex_value =
  //     tex2D<float>(device_acc_disk_tex_.d__tex, mapped_r * cosf(theta) +
  //     0.5f,
  //                  mapped_r * sinf(theta) + 0.5f);
  float const u{theta / (2.0f * M_PIf32) + 0.5f};
  float const v{(r_xy - config_.inner_disk_radius) /
                (config_.outer_disk_radius - config_.inner_disk_radius)};
  tex_value = tex2D<float>(device_acc_disk_tex_.d__tex, u, v);
  if (AccretionDiskTextureType::TRANSMISSION_PROBABILITY ==
          device_acc_disk_tex_.type &&
      tex_value <= 0.5f) {
    return false;
  }
  outer_normal = d_dot_n < 0.0f ? disk_normal_ : -disk_normal_;

  return true;
}

void BlackHole::UploadTexture(
    HostAccretionDiskTexture const host_acc_disk_tex) {
  cudaChannelFormatDesc const ch_desc{cudaCreateChannelDesc(
      sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat)};
  CUDA_CHECK(cudaMallocArray(&device_acc_disk_tex_.d__tex_arr, &ch_desc,
                             host_acc_disk_tex.cols, host_acc_disk_tex.rows));
  CUDA_CHECK(cudaMemcpy2DToArray(
      device_acc_disk_tex_.d__tex_arr, 0UL, 0UL, host_acc_disk_tex.h__texture,
      host_acc_disk_tex.cols * sizeof(float),
      host_acc_disk_tex.cols * sizeof(float), host_acc_disk_tex.rows,
      cudaMemcpyHostToDevice));
  cudaResourceDesc res_desc{};
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypeArray;
  res_desc.res.array.array = device_acc_disk_tex_.d__tex_arr;
  cudaTextureDesc tex_desc{};
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0] = cudaAddressModeClamp;
  tex_desc.addressMode[1] = cudaAddressModeClamp;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.normalizedCoords = 1;
  cudaCreateTextureObject(&device_acc_disk_tex_.d__tex, &res_desc, &tex_desc,
                          NULL);
  device_acc_disk_tex_.type = host_acc_disk_tex.type;
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
  Release();

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

    obj_mem_info_.spheres = MaterialInfoVec(sphere_material_info);

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

HOST_DEVICE_FUNC SchwarzschildGeodesicFunctor::SchwarzschildGeodesicFunctor(
    Config const &config) noexcept
    : config_{config} {
  // assert(config_.Check());
}

HOST_DEVICE_FUNC SchwarzschildGeodesicFunctor::State
SchwarzschildGeodesicFunctor::operator()(float const t,
                                         State const &x0) const noexcept {
  State dx{};
  dx.x() = x0(3);
  dx.y() = x0(4);
  dx.z() = x0(5);
  math::Vector3f const r{x0.x() - config_.position.x(),
                         x0.y() - config_.position.y(),
                         x0.z() - config_.position.z()};
  float const inv_r5{1.0f / powf(r.SquaredNorm(), 2.0f)};
  float const force{-1.5f * config_.sr * am2_ * inv_r5};
  math::Vector3f const rn{r.Normalized()};
  dx(3) = force * rn.x();
  dx(4) = force * rn.y();
  dx(5) = force * rn.z();

  return dx;
}

HOST_DEVICE_FUNC void SchwarzschildGeodesicFunctor::SetAngularMomentum(
    Ray const &ray) noexcept {
  math::Vector3f const r{ray.getOrigin() - config_.position};
  am2_ = r.Cross(ray.getDirection().Normalized()).SquaredNorm();
}

void HostSchwarzschildSpace::Release() {
  for (auto &obj : objects_.black_holes) {
    obj.Release();
  }
  objects_.black_holes.clear();
}

SchwarzschildSpaceCameraPrior HostSchwarzschildSpace::GetCameraPrior(
    math::Vector3f const &camera_position,
    math::Quaternionf const &camera_rotation) const noexcept {
  SchwarzschildSpaceCameraPrior camera_prior{};
  camera_prior.position = camera_position;
  camera_prior.rotation = camera_rotation;

  if (!objects_.black_holes.empty()) {
    camera_prior.nearest_obj_boundary_dist =
        objects_.black_holes[0].GetDistToSphereBoundary(camera_position);
    for (uint64_t i{1UL}; i < objects_.black_holes.size(); ++i) {
      float const dist{
          objects_.black_holes[i].GetDistToSphereBoundary(camera_position)};
      if (dist < camera_prior.nearest_obj_boundary_dist) {
        camera_prior.nearest_obj_boundary_dist = dist;
      }
    }
  }

  return camera_prior;
}

bool DeviceSchwarzschildSpace::Upload(HostSchwarzschildSpace const &host) {
  auto const &objects{host.objects()};
  Release();

  //* Upload blackholes
  {
    uint64_t const num_objs{objects.black_holes.size()};
    uint64_t const blackhole_buffer_size{sizeof(BlackHole) * num_objs};
    CUDA_CHECK(
        cudaMalloc((void **)&objects_.d__black_holes, blackhole_buffer_size));
    CUDA_CHECK(cudaMemcpy(objects_.d__black_holes, objects.black_holes.data(),
                          blackhole_buffer_size, cudaMemcpyHostToDevice));

    std::vector<SchwarzschildGeodesicFunctor> geodesics{};
    geodesics.reserve(num_objs);
    for (uint32_t i{0U}; i < num_objs; ++i) {
      SchwarzschildGeodesicFunctor::Config geodesic_cfg{};
      geodesic_cfg.sr = objects.black_holes.at(i).config().sr;
      geodesic_cfg.position = objects.black_holes.at(i).config().position;
      geodesics.emplace_back(geodesic_cfg);
    }
    uint64_t const geodesic_buffer_size{sizeof(SchwarzschildGeodesicFunctor) *
                                        num_objs};
    CUDA_CHECK(
        cudaMalloc((void **)&objects_.d__geodesics, geodesic_buffer_size));
    CUDA_CHECK(cudaMemcpy(objects_.d__geodesics, geodesics.data(),
                          geodesic_buffer_size, cudaMemcpyHostToDevice));

    objects_.num_black_holes = num_objs;
  }

  return true;
}

void DeviceSchwarzschildSpace::Release() {
  if (nullptr != objects_.d__black_holes) {
    CUDA_CHECK(cudaFree(objects_.d__black_holes));
    objects_.d__black_holes = nullptr;
  }
  if (nullptr != objects_.d__geodesics) {
    CUDA_CHECK(cudaFree(objects_.d__geodesics));
    objects_.d__geodesics = nullptr;
  }
  objects_.num_black_holes = 0UL;
}

DEVICE_FUNC bool DeviceSchwarzschildSpace::Hit(
    Ray const &ray, Intervalf const &interval,
    HitRecordCuda &hit_record) const noexcept {
  if (0UL == objects_.num_black_holes) {
    hit_record.color.SetZero();
    hit_record.final_dir = ray.getDirection();

    return false;
  }

  math::Rk4OdeSolver<float, 6U, SchwarzschildGeodesicFunctor> ode{};
  float s{0.0f};
  float step{0.01f};
  float ds{};
  math::Vector6f x{ray.getOrigin().x(), ray.getOrigin().y(),
                   ray.getOrigin().z()};
  x(3) = ray.getDirection().x();
  x(4) = ray.getDirection().y();
  x(5) = ray.getDirection().z();

  math::Vector6f dx{0.0f, 0.0f, 0.0f};
  math::Vector3f acc{0.0f, 0.0f, 0.0f};
  Ray curr_ray{};
  for (float t{0.0f}; t <= config_.max_marching_time && s <= interval.max();
       t += step) {
    dx.SetZero();
    acc.SetZero();
    bool too_far{true};
    for (uint32_t i{0U}; i < objects_.num_black_holes; ++i) {
      float const dist2{(math::Vector3f{x.x(), x.y(), x.z()} -
                         objects_.d__black_holes[i].config().position)
                            .SquaredNorm()};
      if (dist2 > objects_.d__black_holes[i].config().max_eff_range *
                      objects_.d__black_holes[i].config().max_eff_range) {
        continue;
      }
      SchwarzschildGeodesicFunctor geodesic{objects_.d__geodesics[i]};
      geodesic.SetAngularMomentum(ray);
      too_far = false;
      math::Vector6f dx_i{};
      ode.Step(geodesic, x, t, step, dx_i);
      dx += dx_i;
    }
    if (too_far) {
      break;
    }
    dx /= objects_.num_black_holes;
    ds = utils::norm_3d(dx.data());
    curr_ray.setOrigin(math::Vector3f{x.x(), x.y(), x.z()});
    curr_ray.setDirection(math::Vector3f{dx.x(), dx.y(), dx.z()});
    Intervalf const curr_int{fmax(interval.min() - s, 0.0f),
                             fmin(interval.max() - s, ds)};
    if (s + ds < camera_prior_.nearest_obj_boundary_dist) {
      s += ds;
      x += dx;
      continue;
    }
    HitRecordCuda curr_hit_record{};
    if (HitBlackHoles(curr_ray, curr_int, curr_hit_record)) {
      hit_record.color = curr_hit_record.color;
      hit_record.emitted = curr_hit_record.emitted;
      hit_record.scattered = curr_hit_record.scattered;
      hit_record.final_dir = curr_ray.getDirection();
      hit_record.front_face = curr_hit_record.front_face;
      hit_record.material = curr_hit_record.material;
      hit_record.material_property = curr_hit_record.material_property;
      hit_record.normal = curr_hit_record.normal;
      hit_record.p = curr_hit_record.p;
      hit_record.t = s + curr_hit_record.t;

      return true;
    }
    s += ds;
    x += dx;
  }

  hit_record.color.SetZero();
  hit_record.final_dir = curr_ray.getDirection();

  return false;
}

INLINE_DEVICE_FUNC math::Vector3f get_accel(float const h2,
                                            math::Vector3f const r) {
  return -1.5f * h2 * powf(r.SquaredNorm(), -2.5f) * r;
}

DEVICE_FUNC bool DeviceSchwarzschildSpace::Hit(
    Ray const &ray, Intervalf const &interval,
    float const *const __restrict__ r_list, float *const smem_h2_map,
    HitRecordCuda &hit_record) const noexcept {
  __syncthreads();
  uint32_t const tidx{threadIdx.x + threadIdx.y * blockDim.x};
  float h[3]{};
  for (uint64_t i{0UL}; i < objects_.num_black_holes; ++i) {
    utils::cross_product(r_list + i * 4UL, ray.getDirection().data(), h);
    smem_h2_map[tidx * objects_.num_black_holes + i] = utils::norm_3d(h);
  }

  __syncthreads();

  if (0UL == objects_.num_black_holes) {
    hit_record.color.SetZero();
    hit_record.final_dir = ray.getDirection();

    return false;
  }

  float s{0.0f};
  float step{0.1f};
  float ds{};

  math::Vector3f acc{0.0f, 0.0f, 0.0f};
  math::Vector3f dp{};
  math::Vector3f curr_pos{ray.getOrigin()};
  math::Vector3f curr_dir{ray.getDirection()};
  Ray curr_ray{};
  float const *const h2_list{smem_h2_map + tidx * objects_.num_black_holes};
  math::Vector3f emitted{0.0f, 0.0f, 0.0f};
  math::Vector3f scattered{1.0f, 1.0f, 1.0f};
  bool hit_anything{false};
  for (float t{0.0f}; t <= config_.max_marching_time && s <= interval.max();
       t += step) {
    acc.SetZero();
    for (uint32_t i{0U}; i < objects_.num_black_holes; ++i) {
      acc += get_accel(h2_list[i],
                       curr_pos - objects_.d__black_holes[i].config().position);
    }
    curr_dir += acc * step;
    dp = curr_dir * step;
    curr_pos += dp;
    ds = dp.Norm();
    curr_ray.setOrigin(curr_pos);
    curr_ray.setDirection(curr_dir);
    Intervalf const curr_int{fmax(interval.min() - s, 0.0f),
                             fmin(interval.max() - s, ds)};
    if (s + ds < camera_prior_.nearest_obj_boundary_dist) {
      s += ds;
      continue;
    }
    HitRecordCuda curr_hit_record{};
    if (HitBlackHoles(curr_ray, curr_int, curr_hit_record)) {
      hit_anything = true;
      emitted += scattered.CwiseProduct(curr_hit_record.emitted);
      scattered = scattered.CwiseProduct(curr_hit_record.color);

      if (curr_hit_record.is_event_horizon) {
        hit_record.color = scattered;
        hit_record.emitted = emitted;
        hit_record.final_dir = curr_ray.getDirection();
        hit_record.p = curr_hit_record.p;
        hit_record.t = s + curr_hit_record.t;
        hit_record.is_event_horizon = curr_hit_record.is_event_horizon;

        return true;
      }
    }
    s += ds;
  }

  hit_record.color = scattered;
  hit_record.emitted = emitted;
  hit_record.final_dir = curr_ray.getDirection();
  hit_record.p = curr_ray.getOrigin();
  hit_record.t = s;
  hit_record.is_event_horizon = false;

  return hit_anything;
}

DEVICE_FUNC bool DeviceSchwarzschildSpace::HitBlackHoles(
    Ray const &ray, Intervalf const &interval,
    HitRecordCuda &hit_record) const noexcept {
  HitRecordCuda temp_rec;
  bool hit_anything = false;
  float closest_so_far = interval.max();

  //* Hit blackholes
  for (uint64_t i{0UL}; i < objects_.num_black_holes; ++i) {
    auto const object{objects_.d__black_holes + i};
    if (object->Hit(ray, Intervalf(interval.min(), closest_so_far), temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      hit_record = temp_rec;
    }
  }

  return hit_anything;
}

}  // namespace cuda