#include "Camera.h"
#include "Sphere.h"
#include "HittableList.h"  
#include "RayTracer.h" 
#include "Lambertian.h" 
#include "Metal.h" 
#include "Dielectric.h" 
#include "SimpleSkyBackground.h" 

Eigen::Quaterniond getCameraRotation(const Eigen::Vector3d &lookfrom, 
                                     const Eigen::Vector3d &lookat, 
                                     const Eigen::Vector3d &vup) {
    Eigen::Matrix3d R; 
    R.col(2) = (lookat - lookfrom).normalized(); 
    R.col(0) = -(vup.cross(R.col(2))).normalized(); 
    R.col(1) = R.col(2).cross(R.col(0)); 
    return Eigen::Quaterniond{R}; 
}

int main(int argc, char **argv) {
    RandomNumberGenerator rng; 

    HittableList world; 

    const MaterialBase::Ptr ground_material = 
        std::make_shared<Lambertian>(Eigen::Vector3d{0.5, 0.5, 0.5}); 
    world.add(std::make_shared<Sphere>(Point3D{0, -1000, 0}, 1000, ground_material)); 

    for (int a = -11; a < 11; ++a) {
        for (int b = -11; b < 11; ++b) {
            const double choose_material = rng.uniform01(); 
            const Point3D center{a + 0.9 * rng.uniform01(), 
                                 0.2, 
                                 b + 0.9 * rng.uniform01()}; 
            
            if ((center - Point3D{4.0, 0.2, 0.0}).norm() > 0.9) {
                MaterialBase::Ptr material; 
                if (choose_material < 0.8) {
                    //* Diffuse 
                    const Eigen::Vector3d albedo = 
                        rng.uniformRealMatrix<3, 1>().cwiseProduct(rng.uniformRealMatrix<3, 1>()); 
                    material = std::make_shared<Lambertian>(albedo); 
                    world.add(std::make_shared<Sphere>(center, 0.2, material)); 
                } else if (choose_material < -0.95) {
                    //* Meta 
                    const Eigen::Vector3d albedo = rng.uniformRealMatrix<3, 1>(0.5, 1.0); 
                    const double fuzz = rng.uniformReal(0.0, 0.5); 
                    material = std::make_shared<Metal>(albedo, fuzz); 
                    world.add(std::make_shared<Sphere>(center, 0.2, material)); 
                } else {
                    //* Glass 
                    material = std::make_shared<Dielectric>(1.5); 
                    world.add(std::make_shared<Sphere>(center, 0.2, material)); 
                }
            }
        }
    } 

    const MaterialBase::Ptr material1 = std::make_shared<Dielectric>(1.5); 
    world.add(std::make_shared<Sphere>(Point3D{0, 1, 0}, 1.0, material1)); 

    const MaterialBase::Ptr material2 = std::make_shared<Lambertian>(Eigen::Vector3d{0.4, 0.2, 0.1}); 
    world.add(std::make_shared<Sphere>(Point3D{-4, 1, 0}, 1.0, material2)); 

    const MaterialBase::Ptr material3 = std::make_shared<Metal>(Eigen::Vector3d{0.7, 0.6, 0.5}, 0.0); 
    world.add(std::make_shared<Sphere>(Point3D{4, 1, 0}, 1.0, material3)); 

    const double aspect_ratio = 16.0 / 9.0; 
    const size_t image_width = 1200; 
    const size_t image_height = static_cast<size_t>(image_width / aspect_ratio); 
    const double vfov = 20.0  * M_PI / 180.0; 
    const double focal_length = 10.0; 
    const double fov_height = 2.0 * std::tan(vfov / 2.0) * focal_length; 
    const double fov_width = fov_height * (static_cast<double>(image_width) / image_height); 
    const double defocus_angle = 0.6 * M_PI / 180.0; 
    const Point3D camera_position{13, 2, 3}; 
    const Point3D lookat{0.0, 0.0, 0.0}; 
    const Eigen::Vector3d vup{0.0, 1.0, 0.0}; 
    const Eigen::Quaterniond camera_rotation = getCameraRotation(camera_position, lookat, vup); 

    Camera camera(image_width, image_height, 
                  fov_width, fov_height, 
                  focal_length, 
                  defocus_angle, 
                  camera_position, 
                  camera_rotation); 
    
    RayTracer::Config renderer_config; 
    renderer_config.background = std::make_shared<SimpleSkyBackground>(); 
    renderer_config.samples_per_pixel = 500; 
    renderer_config.max_depth = 50; 
    renderer_config.max_sample_pert = 0.5; 
    RayTracer renderer(renderer_config); 
    TicToc tictoc; 
    tictoc.tic(); 
    renderer.renderMultiThread(camera, world, 8, std::cout); 
    const double rendering_time = tictoc.toc(); 
    std::clog << "Rendering time: " 
              << int(rendering_time) / 3600 << ":" 
              << (int(rendering_time) % 3600) / 60 << ":" 
              << (rendering_time - int(rendering_time) / 60 * 60) << "." 
              << std::endl; 

    return 0; 
}