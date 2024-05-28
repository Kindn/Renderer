/*
 * filename: HittableList.h 
 * author:   Peiyan Liu, HITSZ
 * E-mail:   1434615509@qq.com
 * brief:    
 */

#ifndef _HITTABLE_LIST_H_
#define _HITTABLE_LIST_H_ 

#include "HittableBase.h"

class HittableList : public HittableBase {
public: 
    HittableList(const MaterialBase::Ptr &material = nullptr): 
    HittableBase(material) {

    } 

    HittableList(HittableBase::Ptr object, 
                 const MaterialBase::Ptr &material = nullptr): 
    HittableBase(material) {

    } 

    void clear() {
        objects_.clear(); 
    } 

    void add(HittableBase::Ptr object) {
        objects_.push_back(object); 
    } 

    HittableBase::Ptr &operator [] (const size_t &idx) {
        return objects_[idx]; 
    } 

    const HittableBase::Ptr &operator [] (const size_t &idx) const {
        return objects_[idx]; 
    } 

    virtual bool hit(const Ray &ray, 
                     const Intervald &interval, 
                     HitRecord &hit_record) const override; 

private: 
    std::vector<HittableBase::Ptr> objects_; 
}; 

#endif // _HITTABLE_LIST_H_