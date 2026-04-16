cmake_minimum_required(VERSION 3.12)

project(renderer LANGUAGES CXX)

set(USE_CUDA ON)
set(VISUALIZE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3 -std=c++14 -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -march=native")

set(EIGEN3_INCLUDE_DIRS "/usr/include/eigen3")

set(OpenCV_DIR "/home/lpy/opencv-4.5.0-cuda/opencv-4.5.0/build/install/lib/cmake/opencv4")
find_package(OpenCV 4 REQUIRED)

include_directories(
    src
    ${EIGEN3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

if(USE_CUDA)
    # add_compile_definitions(USE_CUDA=1)
    add_definitions(-DUSE_CUDA)
    find_package(CUDA 11.3 REQUIRED)

    if(CUDA_FOUND)
        message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
        message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
        message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")

        include_directories(${CUDA_INCLUDE_DIRS})
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -gencode arch=compute_75,code=sm_75 --ptxas-options=-v --maxrregcount=64 -lineinfo")
        set(CMAKE_CUDA_ARCHITECTURES 75-real)
        set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc)

        enable_language(CUDA)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
        set(PROJECT_LANGUAGE CXX CUDA)
        message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
        message(STATUS "PROJECT_LANGUAGE: ${PROJECT_LANGUAGE}")

        set(CUB_DIR "/usr/local/cuda-11.3/targets/x86_64-linux/include/cub/cmake/")
        set(Thrust_DIR "/usr/local/cuda-11.3/targets/x86_64-linux/include/thrust/cmake/")
        find_package(Thrust REQUIRED CONFIG) # CMake 会根据你手动指定的 Thrust_DIR 来查找
        thrust_create_target(Thrust)
        include_directories(${Thrust_INCLUDE_DIRS})
        message(STATUS "Thrust_INCLUDE_DIRS: ${Thrust_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "Cannot find CUDA. ")
    endif()
else()
    set(PROJECT_LANGUAGE CXX)
endif()

if(VISUALIZE)
    add_definitions(-DVISUALIZE)
endif()

set(RENDERER_SRC
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
set(RENDERER_DEPEND_LIBS
    ${OpenCV_LIBRARIES}
)

if(USE_CUDA)
    set(RENDERER_SRC ${RENDERER_SRC}
        src/cuda/materials.cu
        src/cuda/hittables.cu
        src/cuda/bhlite.cu
        src/cuda/camera.cu
        src/cuda/rendering.cu
        src/cuda/background.cu
        src/cuda/post_processing.cu
        src/cuda/utils/common.cpp
    )
    set(RENDERER_DEPEND_LIBS ${RENDERER_DEPEND_LIBS}
        ${CUDA_LIBRARIES}
    )
endif()

message(STATUS "RENDERER_SRC: ${RENDERER_SRC}")

add_library(renderer SHARED ${RENDERER_SRC})
target_link_libraries(renderer ${RENDERER_DEPEND_LIBS} ${OpenCV_LIBRARIES})

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

add_executable(rtweekend_cuda rtweekend_cuda.cpp)
target_link_libraries(rtweekend_cuda renderer ${RENDERER_DEPEND_LIBS} ${OpenCV_LIBRARIES})

add_executable(video_generator video_generator.cpp)
target_link_libraries(video_generator renderer ${RENDERER_DEPEND_LIBS} ${OpenCV_LIBRARIES})

add_executable(blackhole_rendering blackhole_rendering.cpp)
target_link_libraries(blackhole_rendering renderer ${RENDERER_DEPEND_LIBS} ${OpenCV_LIBRARIES})

add_executable(realtime_blackhole_rendering realtime_blackhole_rendering.cpp)
target_link_libraries(realtime_blackhole_rendering renderer ${RENDERER_DEPEND_LIBS} ${OpenCV_LIBRARIES})
