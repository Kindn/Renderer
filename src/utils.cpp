/*
 * filename: utils.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "utils.h"

extern const float inff = std::numeric_limits<float>::infinity();
extern const float epsf = std::numeric_limits<float>::epsilon();

void writeColorToOStream(std::ostream &out, const Color &color) {
  out << int(color.x()) << ' ' << int(color.y()) << ' ' << int(color.z())
      << ' ';
}

void writeColorToOStream(std::ostream &out, const math::Vector3f &color) {
  static const Intervalf intensity(0.000, 0.999);
  out << int(256 * intensity.clamp(linearToGamma(color.x(), 2.0f))) << ' '
      << int(256 * intensity.clamp(linearToGamma(color.y(), 2.0f))) << ' '
      << int(256 * intensity.clamp(linearToGamma(color.z(), 2.0f))) << ' ';
}
