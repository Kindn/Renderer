/*
 * filename: RayTracer.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#ifndef _RAY_TRACER_H_
#define _RAY_TRACER_H_

#include <functional>

#include "HittableList.h"
#include "MaterialBase.h" 
#include "BackgroundBase.h"
#include "Camera.h"

class RayTracer {
public: 
    struct Config {
        BackgroundBase::Ptr background; 
        int max_depth{50}; 
        int samples_per_pixel{100}; 
        double max_sample_pert{0.5}; 

        bool check() const {
            return (background != nullptr) && 
                   (max_depth > 0) && 
                   (samples_per_pixel > 0); 
        }
    }; 

    RayTracer(const Config &config): 
    config_{config} {
        assert(config_.check()); 
    } 

public: 
    Eigen::Vector3d getRayColor(const Ray &ray, 
                                const HittableList &world, 
                                const int &depth) const; 
    
    void render(const Camera &camera, 
                const HittableList &world, 
                std::ostream &out_stream); 

    void renderMultiThread(const Camera &camera, 
                           const HittableList &world, 
                           const size_t &max_num_threads, 
                           std::ostream &out_stream); 
    
    void threadRendering(const Camera *camera, 
                         const HittableList *world, 
                         const size_t &start_row, 
                         const size_t &num_rows, 
                         std::vector<std::vector<Eigen::Vector3d>> *row_buffers, 
                         const size_t &idx); 

    void threadPrintingProgressBar(const size_t &image_height) {
        size_t num_finished_rows; 
        const int bar_length = 100;
        do {
            num_finished_rows = nums_finished_rows_.sum(); 
            const double progress = 
                double(num_finished_rows) / image_height; 
            std::clog << "\rRendering: ["; 
            for (int i = 0; i < progress * bar_length; ++i) {
                std::clog << "*"; 
            }
            for (int i = progress * bar_length; i < bar_length; ++i) {
                std::clog << " "; 
            }
            std::clog << "] " << 100 * progress << "%" << std::flush; 
        } while (num_finished_rows < image_height); 
        std::clog << std::endl; 
    }


private: 
    Config config_; 

    std::mutex mutex_; 
    Eigen::Matrix<size_t, Eigen::Dynamic, 1> nums_finished_rows_; 
}; 

#endif // _RAY_TRACER_H_
