/*
 * filename: RayTracer.cpp 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */ 

#include "RayTracer.h" 

Eigen::Vector3d RayTracer::getRayColor(const Ray &ray, 
                                       const HittableList &world, 
                                       const int &depth) const {
    if (depth <= 0) {
        return Eigen::Vector3d::Zero(); 
    }

    HitRecord hit_record; 
    if (world.hit(ray, Intervald{0.001, infd}, hit_record)) {
        if (hit_record.material != nullptr) {
            Ray scattered; 
            Eigen::Vector3d attenuation; 
            if (hit_record.material->scatter(ray, hit_record, attenuation, scattered)) {
                return attenuation.cwiseProduct(getRayColor(scattered, world, depth - 1));
            } else {
                return Eigen::Vector3d::Zero(); 
            }
        } 
    }

    return config_.background->getRayColor(ray); 
}

void RayTracer::render(const Camera &camera, 
                       const HittableList &world, 
                       std::ostream &out_stream) {
    const size_t &image_width = camera.getImageWidth(); 
    const size_t &image_height = camera.getImageHeight(); 

    out_stream << "P3\n" << image_width << " " << image_height << "\n255\n"; 
    for (size_t row = 0; row < image_height; ++row) {
        std::clog << "\rScanlines remaining: " << (image_height - row) << " " << std::flush; 
        for (size_t col = 0; col < image_width; ++col) {
            Eigen::Vector3d color{0.0, 0.0, 0.0}; 
            for (int i = 0; i < config_.samples_per_pixel; ++i) {
                const Ray ray = camera.getDefocusPerturbedRay(PixCoord(col, row), config_.max_sample_pert); 
                color += getRayColor(ray, world, config_.max_depth); 
            }
            color /= double(config_.samples_per_pixel);
            writeColorToOStream(out_stream, color); 
            out_stream << std::endl; 
        }
    }
    std::clog << "\nDone.                  " << std::endl; 
}

void RayTracer::renderMultiThread(const Camera &camera, 
                                  const HittableList &world, 
                                  const size_t &max_num_threads, 
                                  std::ostream &out_stream) {
    assert(max_num_threads > 0); 
    const size_t &image_width = camera.getImageWidth(); 
    const size_t &image_height = camera.getImageHeight(); 

    const size_t kMinNumRowsPerThread = 3; 
    const size_t rows_per_thread = std::max(image_height / max_num_threads, kMinNumRowsPerThread); 
    const size_t kNumThreads = std::ceil(image_height / rows_per_thread); 
    std::clog << kNumThreads << " threads will be created for rendering. " << std::endl; 

    nums_finished_rows_.resize(kNumThreads); 
    nums_finished_rows_.setZero(); 
    std::vector<std::vector<Eigen::Vector3d>> row_buffers{image_height}; 
    for (size_t i = 0; i < image_height; ++i) {
        row_buffers[i].resize(image_width);
    }
    std::vector<ThreadGuard*> thread_guards; 
    thread_guards.reserve(kNumThreads + 1); 
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads + 1); 
    size_t curr_row = 0;  
    for (size_t i = 0; i < kNumThreads; ++i) {
        const size_t num_rows = 
            (i == kNumThreads - 1) ? (image_height - rows_per_thread * (kNumThreads - 1)) : rows_per_thread; 
        threads.emplace_back(&RayTracer::threadRendering, 
            this, &camera, &world, curr_row, num_rows, &row_buffers, i); 
        thread_guards.push_back(new ThreadGuard(threads.back())); 
        curr_row += num_rows; 
    }
    threads.emplace_back(&RayTracer::threadPrintingProgressBar, 
        this, image_height);
    thread_guards.push_back(new ThreadGuard(threads.back())); 

    for (size_t i = 0; i < thread_guards.size(); ++i) {
        delete thread_guards[i]; 
        thread_guards[i] = nullptr; 
    }

    out_stream << "P3\n" << image_width << " " << image_height << "\n255\n"; 
    for (size_t row = 0; row < image_height; ++row) {
        for (size_t col = 0; col < image_width; ++col) {
            writeColorToOStream(out_stream, row_buffers[row][col]); 
        }
    }
}

void RayTracer::threadRendering(const Camera *camera, 
                                const HittableList *world, 
                                const size_t &start_row, 
                                const size_t &num_rows, 
                                std::vector<std::vector<Eigen::Vector3d>> *row_buffers, 
                                const size_t &idx) {
    const size_t end_row = std::min(start_row + num_rows - 1, row_buffers->size() - 1); 
    for (size_t row = start_row; row <= end_row; ++row) {
        std::vector<Eigen::Vector3d> &row_buffer = row_buffers->at(row); 
        for (size_t col = 0; col < row_buffer.size(); ++col) {
            Eigen::Vector3d &color = row_buffer[col]; 
            color.setZero(); 
            for (int i = 0; i < config_.samples_per_pixel; ++i) {
                const Ray ray = camera->getDefocusPerturbedRay(PixCoord(col, row), config_.max_sample_pert); 
                color += getRayColor(ray, *world, config_.max_depth); 
            }
            color /= double(config_.samples_per_pixel);
        }
        mutex_.lock(); 
        ++nums_finished_rows_(idx); 
        mutex_.unlock(); 
    }
}
