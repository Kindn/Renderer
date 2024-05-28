#include "Camera.h"
#include "Sphere.h"
#include "HittableList.h"  
#include "RayTracer.h" 
#include "Lambertian.h" 
#include "Metal.h" 
#include "Dielectric.h" 
#include "SimpleSkyBackground.h" 

int main(int argc, char **argv) {
    HittableList world; 

    MaterialBase::Ptr material_ground = std::make_shared<Lambertian>(Eigen::Vector3d{0.8, 0.8, 0.0}); 
    MaterialBase::Ptr material_center = std::make_shared<Lambertian>(Eigen::Vector3d{0.1, 0.2, 0.5}); 
    MaterialBase::Ptr material_left = std::make_shared<Dielectric>(1.50); 
    MaterialBase::Ptr material_bubble = std::make_shared<Dielectric>(1.00 / 1.50); 
    MaterialBase::Ptr material_right = std::make_shared<Metal>(Eigen::Vector3d{0.8, 0.6, 0.2}, 1.0); 

    BackgroundBase::Ptr background = std::make_shared<SimpleSkyBackground>(); 

    world.add(std::make_shared<Sphere>(Point3D{0.0, 0.0, 2.7}, 1.5, material_center)); 
    world.add(std::make_shared<Sphere>(Point3D{0.0, 101.5, 2.0}, 100, material_ground)); 
    world.add(std::make_shared<Sphere>(Point3D{-3.0, 0, 2.0}, 1.5, material_left)); 
    world.add(std::make_shared<Sphere>(Point3D{-3.0, 0, 2.0}, 1.2, material_bubble)); 
    world.add(std::make_shared<Sphere>(Point3D{3.0, 0, 2.0}, 1.5, material_right)); 

    const double aspect_ratio = 16.0 / 9.0; 
    const size_t image_width = 400; 
    const size_t image_height = static_cast<size_t>(image_width / aspect_ratio); 
    const double fov_height = 0.5; 
    const double fov_width = fov_height * (static_cast<double>(image_width) / image_height); 
    const double focal_length = 1.0; 
    const double defocus_angle = 10.0 * M_PI / 180.0; 
    const Point3D camera_position(-4, -4, -2); 
    const Eigen::Vector3d camera_rotation_rpy{-std::atan2(1.0, std::sqrt(2.0)), 
                                              M_PI / 4.0, 
                                              0.0}; 
    const Eigen::Quaterniond camera_rotation = 
        Eigen::Quaterniond{std::cos(camera_rotation_rpy.z() / 2.0), 0, 0, std::sin(camera_rotation_rpy.z() / 2.0)} * 
        Eigen::Quaterniond{std::cos(camera_rotation_rpy.y() / 2.0), 0, std::sin(camera_rotation_rpy.y() / 2.0), 0} * 
        Eigen::Quaterniond{std::cos(camera_rotation_rpy.x() / 2.0), std::sin(camera_rotation_rpy.x() / 2.0), 0, 0}; 

    const Camera camera(image_width, image_height, 
                        fov_width, fov_height, 
                        focal_length, 
                        defocus_angle, 
                        camera_position, 
                        camera_rotation); 
    
    RayTracer::Config renderer_config; 
    renderer_config.background = background;
    renderer_config.samples_per_pixel = 20;  
    RayTracer renderer(renderer_config); 
    renderer.render(camera, world, std::cout); 

    return 0; 
}
