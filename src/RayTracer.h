/*
 * filename: RayTracer.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _RAY_TRACER_H_
#define _RAY_TRACER_H_

#include <functional>

#include <opencv2/opencv.hpp>

#include "BackgroundBase.h"
#include "Camera.h"
#include "HittableList.h"
#include "MaterialBase.h"

class RayTracer {
public:
  struct Config {
    BackgroundBase::Ptr background;
    int max_depth{50};
    int samples_per_pixel{100};
    float max_sample_pert{0.5f};

    bool check() const {
      return (background != nullptr) && (max_depth > 0) &&
             (samples_per_pixel > 0);
    }
  };

  RayTracer(const Config &config) : config_{config} { assert(config_.check()); }

public:
  math::Vector3f getRayColor(const Ray &ray, const HittableList &world,
                             const int &depth) const;

  void render(const Camera &camera, const HittableList &world,
              std::ostream &out_stream);

  void renderMultiThread(const Camera &camera, const HittableList &world,
                         const size_t &max_num_threads,
                         std::ostream &out_stream);

  void threadRendering(const Camera *camera, const HittableList *world,
                       const size_t &start_row, const size_t &num_rows,
                       std::vector<std::vector<math::Vector3f>> *row_buffers,
                       const size_t &idx);

  void threadPrintingProgressBar(
      const size_t &image_height, const size_t &image_width,
      std::vector<std::vector<math::Vector3f>> *row_buffers) {
#ifdef VISUALIZE
    cv::Mat cv_img(image_height, image_width, CV_8UC3, cv::Scalar(0));
#endif
    size_t num_finished_rows;
    const int bar_length = 100;
    const Intervalf intensity(0.000, 0.999);
    do {
      num_finished_rows = std::accumulate(nums_finished_rows_.begin(),
                                          nums_finished_rows_.end(), 0UL);
      const float progress = float(num_finished_rows) / image_height;
      std::clog << "\rRendering: [";
      for (int i = 0; i < progress * bar_length; ++i) {
        std::clog << "*";
      }
      for (int i = progress * bar_length; i < bar_length; ++i) {
        std::clog << " ";
      }
      std::clog << "] " << 100 * progress << "%" << std::flush;
#ifdef VISUALIZE
      for (size_t row = 0; row < image_height; ++row) {
        for (size_t col = 0; col < image_width; ++col) {
          cv_img.at<cv::Vec3b>(row, col)[0] =
              uint8_t(256 * intensity.clamp(linearToGamma(
                                row_buffers->at(row)[col].z(), 2.0f)));
          cv_img.at<cv::Vec3b>(row, col)[1] =
              uint8_t(256 * intensity.clamp(linearToGamma(
                                row_buffers->at(row)[col].y(), 2.0f)));
          cv_img.at<cv::Vec3b>(row, col)[2] =
              uint8_t(256 * intensity.clamp(linearToGamma(
                                row_buffers->at(row)[col].x(), 2.0f)));
        }
      }
      cv::imshow("rtweekend", cv_img);
      cv::waitKey(5);
#endif
    } while (num_finished_rows < image_height);
#ifdef VISUALIZE
    cv::waitKey();
#endif
    std::clog << std::endl;
  }

private:
  Config config_;

  std::mutex mutex_;
  std::vector<uint64_t> nums_finished_rows_;
};

#endif // _RAY_TRACER_H_
