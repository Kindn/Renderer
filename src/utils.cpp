/*
 * filename: utils.cpp 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "utils.h" 

extern const double infd = std::numeric_limits<double>::infinity(); 
extern const double epsd = std::numeric_limits<double>::epsilon(); 

void writeColorToOStream(std::ostream &out, const Color &color) {
    out << int(color.x()) << ' ' 
        << int(color.y()) << ' ' 
        << int(color.z()) << ' '; 
}

void writeColorToOStream(std::ostream &out, const Eigen::Vector3d &color) {
    static const Intervald intensity(0.000, 0.999); 
    out << int(256 * intensity.clamp(linearToGamma(color.x(), 2.0))) << ' ' 
        << int(256 * intensity.clamp(linearToGamma(color.y(), 2.0))) << ' ' 
        << int(256 * intensity.clamp(linearToGamma(color.z(), 2.0))) << ' '; 
}

