/*
 * filename: SimpleSkyBackground.cpp
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#include "SimpleSkyBackground.h"

math::Vector3f SimpleSkyBackground::getRayColor(const Ray &ray) const {
  const math::Vector3f &direction = ray.getDirection();
  const float a = 0.5f * (-direction.y() + 1.0f);
  const math::Vector3f color1(1.0f, 1.0f, 1.0f);
  const math::Vector3f color2(0.5f, 0.7, 1.0f);

  return (1.0f - a) * color1 + a * color2;
}