/*
 * filename: SimpleSkyBackground.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _SIMPLE_SKY_BACKGOUND_H_
#define _SIMPLE_SKY_BACKGOUND_H_

#include "BackgroundBase.h"

class SimpleSkyBackground : public BackgroundBase {
public:
  typedef std::shared_ptr<SimpleSkyBackground> Ptr;

  SimpleSkyBackground() {}
  SimpleSkyBackground(const math::Vector3f &color1,
                      const math::Vector3f &color2)
      : color1_{color1}, color2_{color2} {}

  

public:
  math::Vector3f getRayColor(const Ray &ray) const override;

private:
  math::Vector3f color1_{1.0f, 1.0f, 1.0f};
  math::Vector3f color2_{0.5f, 0.7, 1.0f};
};

#endif