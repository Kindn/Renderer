/*
 * filename: BackgroundBase.h
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:
 */

#ifndef _BACKGROUND_BASE_H_
#define _BACKGROUND_BASE_H_

#include "Ray.h"
#include "utils.h"

class BackgroundBase {
public:
  typedef std::shared_ptr<BackgroundBase> Ptr;

  HOST_DEVICE_FUNC virtual ~BackgroundBase() = default;

  

public:
  virtual math::Vector3f getRayColor(const Ray &ray) const = 0;
};

#endif // _BACKGROUND_BASE_H_
