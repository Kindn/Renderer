/*
 * filename: utils.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _TYPES_H_
#define _TYPES_H_

#include <iostream>
#include <vector>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <mutex>
#include <chrono>

#include <eigen3/Eigen/Dense>

// namespace Eigen {

typedef Eigen::Matrix<uint8_t, 3, 1> Color; 
typedef Eigen::Matrix<size_t, 2, 1> PixCoord; 
typedef Eigen::Vector3d Point3D; 
typedef Eigen::Vector2d Point2D; 

template <typename T>
T inf() {
    return std::numeric_limits<T>::infinity(); 
}

template <typename T> 
class Interval {
public: 
    // static constexpr T inf = std::numeric_limits<T>::infinity(); 

    Interval(const T &min = inf<T>(), 
             const T &max = -inf<T>()): 
    min_{min}, max_{max} {

    }

    T size() const {
        return max_ - min_; 
    } 

    bool isEmpty() const {
        return min_ > max_; 
    }

    bool contains(const T &x) const {
        return x >= min_ && x <= max_; 
    }

    bool surrounds(const T &x) const {
        return x > min_ && x < max_; 
    }

    T& min() {
        return min_; 
    }

    const T& min() const {
        return min_; 
    }

    T& max() {
        return max_; 
    }

    const T& max() const {
        return max_; 
    } 

    T clamp(const T &x) const {
        return (x > max_) ? max_ : ((x < min_) ? min_ : x); 
    }

    static Interval<T> positive() {
        return Interval<T>(0, inf<T>()); 
    }

    static Interval<T> negative() {
        return Interval<T>(-inf<T>(), 0); 
    }

    static Interval<T> all() {
        return Interval<T>(-inf<T>(), inf<T>()); 
    }

private: 
    T min_; 
    T max_; 
}; 

typedef Interval<double> Intervald; 
typedef Interval<int>    Intervali;

extern const double infd; 
extern const double epsd; 

class RandomNumberGenerator {
public: 
    RandomNumberGenerator(): 
    seed_(time(NULL)), engine_(seed_)
    {}

    RandomNumberGenerator(const std::uint_fast32_t &seed): 
    seed_(seed), engine_(seed_)
    {}

    ~RandomNumberGenerator() 
    {}

public: 
    void setSeed(const std::uint_fast32_t &seed) {
        seed_ = seed;
        engine_.seed(seed_);
    }

    double uniform01() {
        return urd01_(engine_);
    }

    double uniformReal(const double &lower, const double &upper) {
        assert(lower <= upper);
        return lower + (upper - lower) * urd01_(engine_);
    }

    int uniformInteger(const int &lower, const int &upper) {
        assert(lower <= upper);
        std::uniform_int_distribution<int> uid(lower, upper);
        return uid(engine_);
    }

    size_t uniformIndex(const size_t &lower, const size_t &upper) {
        assert(lower <= upper);
        std::uniform_int_distribution<size_t> uid(lower, upper);
        return uid(engine_);
    }

    // From: "Uniform Random Rotations", Ken Shoemake, Graphics Gems III,
    //       pg. 124-132
    void uniformQuaternion(double &x, double &y, double &z, double &w) {
        double t = urd01_(engine_);
        double r1 = sqrt(1.0 - t), r2 = sqrt(t);
        double th1 = 2.0 * M_PI * urd01_(engine_);
        double th2 = 2.0 * M_PI * urd01_(engine_);
        double c1 = cos(th1), s1 = sin(th1);
        double c2 = cos(th2), s2 = sin(th2);
        x = s1 * r1;
        y = c1 * r1;
        z = s2 * r2;
        w = c2 * r2;
    } 

    template <size_t Rows, size_t Cols> 
    Eigen::Matrix<double, Rows, Cols> uniformRealMatrix(const double &lower = 0.0, 
                                                        const double &upper = 1.0) {
        const double scale = (lower + upper) / 2.0; 
        Eigen::Matrix<double, Rows, Cols> ret = Eigen::Matrix<double, Rows, Cols>::Random(); 
        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                ret(r, c) = lower + (ret(r, c) + 1) * scale; 
            }
        }
        return ret; 
    }

    Point3D uniformPoint3DOnUnitSphere3D() {
        const double u = uniformReal(0.0, 1.0); 
        const double v = uniformReal(-1.0, 1.0); 
        const double phi = 2.0 * M_PI * u; 
        const double theta = std::acos(v); 
        return Point3D{std::sin(theta) * std::cos(phi), 
                       std::sin(theta) * std::sin(phi), 
                       std::cos(theta)}; 
    }

    Point3D uniformPoint3DOnUnitHemiSphere3D(const Eigen::Vector3d &normal) {
        const Point3D pr = uniformPoint3DOnUnitSphere3D(); 
        return (pr.dot(normal) > 0.0) ? pr : -pr; 
    } 

    Point2D uniformPoint2DInUnitCircle() {
        const double r = std::sqrt(uniform01()); 
        const double theta = 2.0 * M_PI * uniform01(); 
        return Point2D{r * std::cos(theta), 
                       r * std::sin(theta)};
    }

private:
    std::uint_fast32_t seed_;
    std::mt19937 engine_;
    std::uniform_real_distribution<double> urd01_{0.0, 1.0};
    std::normal_distribution<double> nd_{0.0, 1.0}; 
}; 

class ThreadGuard
 {
 private:
     std::thread& t;
 public:
     explicit ThreadGuard(std::thread& t_):t(t_){}
     
     ~ThreadGuard(){
         if(t.joinable()) { 
             t.join(); 
         }
     }
 };

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::steady_clock::now();
    }

    double toc()
    {
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = 
          std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        return elapsed_seconds.count();
    }

  private:
    std::chrono::steady_clock::time_point start, end;
};

inline double linearToGamma(const double &x, 
                            const double &gamma) {
    return (x > 0.0) ? std::pow(x, 1.0 / gamma) : 0.0; 
}

inline Eigen::Vector3d reflect(const Eigen::Vector3d &v, 
                               const Eigen::Vector3d &n) {
    return v - 2.0 * v.dot(n) * n;
} 

inline Eigen::Vector3d refract(const Eigen::Vector3d &uv, 
                               const Eigen::Vector3d &n, 
                               const double &etai_over_etat) {
    const double cos_theta = std::min(-uv.dot(n), 1.0); 
    const Eigen::Vector3d r_out_perp = etai_over_etat * (uv + cos_theta * n); 
    const Eigen::Vector3d r_out_para = 
        -std::sqrt(1.0 - r_out_perp.squaredNorm()) * n; 
    return r_out_perp + r_out_para; 
}

void writeColorToOStream(std::ostream &out, const Color &color); 

void writeColorToOStream(std::ostream &out, const Eigen::Vector3d &color); 

// }

#endif // _TYPES_H_
