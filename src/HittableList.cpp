/*
 * filename: HittableList.cpp 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#include "HittableList.h"

bool HittableList::hit(const Ray &ray, 
                       const Intervald &interval, 
                       HitRecord &hit_record) const {
    HitRecord temp_rec; 
    bool hit_anything = false; 
    double closest_so_far = interval.max(); 
    for (const auto &object : objects_) {
        if (object->hit(ray, Intervald(interval.min(), closest_so_far), temp_rec)) {
            hit_anything = true; 
            closest_so_far = temp_rec.t; 
            hit_record = temp_rec; 
        }
    }

    return hit_anything; 
}
