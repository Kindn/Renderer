cmake_minimum_required(VERSION 2.8.3)
project(renderer)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++17 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -march=native")

set(EIGEN3_INCLUDE_DIRS "/usr/include/eigen3")

include_directories(
    src
    ${EIGEN3_INCLUDE_DIRS}
)

add_library(renderer SHARED
    src/utils.cpp src/utils.h
    src/Ray.cpp src/Ray.h 
    src/Camera.cpp src/Camera.h 
    src/HittableBase.h 
    src/Sphere.cpp src/Sphere.h 
    src/HittableList.cpp src/HittableList.h 
    src/MaterialBase.h
    src/Lambertian.cpp src/Lambertian.h 
    src/Metal.cpp src/Metal.h 
    src/Dielectric.cpp src/Dielectric.h 
    src/BackgroundBase.h 
    src/SimpleSkyBackground.cpp src/SimpleSkyBackground.h 
    src/RayTracer.cpp src/RayTracer.h 
)

add_executable(hello_world hello_world.cpp)

add_executable(simple_rendering_test simple_rendering_test.cpp) 
target_link_libraries(simple_rendering_test renderer) 

add_executable(single_sphere_rendering_test single_sphere_rendering_test.cpp)
target_link_libraries(single_sphere_rendering_test renderer)

add_executable(multiple_object_rendering_test multiple_object_rendering_test.cpp)
target_link_libraries(multiple_object_rendering_test renderer)

add_executable(diffuse_material_test diffuse_material_test.cpp)
target_link_libraries(diffuse_material_test renderer)

add_executable(lambertian_test lambertian_test.cpp)
target_link_libraries(lambertian_test renderer) 

add_executable(metal_test metal_test.cpp) 
target_link_libraries(metal_test renderer) 

add_executable(dielectric_test dielectric_test.cpp) 
target_link_libraries(dielectric_test renderer) 

add_executable(defocus_test defocus_test.cpp) 
target_link_libraries(defocus_test renderer) 

add_executable(rtweekend rtweekend.cpp) 
target_link_libraries(rtweekend renderer)
