#pragma once

#include "math/matrix.h"

#include <fstream>

namespace geometry {

template <typename DType> class TriangleMesh {
public:
  typedef math::Matrix<DType, 3U, 1U> VertexType;
  typedef math::Matrix<DType, 3U, 1U> NormalType;
  typedef math::Matrix<uint8_t, 3U, 1U> ColorType;

  struct Vertex {
    VertexType point{};
    ColorType color{};
  };

  struct Face {
    uint64_t vidx[3]{};
  };

  TriangleMesh() = default;

  void AddVertex(VertexType const &point, ColorType const &color) noexcept {
    vertices_.emplace_back({point, color});
  }

  bool AddFace(uint64_t const vidx[3]) noexcept {
    uint64_t const nv{vertices_.size()};
    if (vidx[0] >= nv || vidx[1] >= nv || vidx[2] >= nv) {
      return false;
    }

    faces_.emplace_back();
    faces_.back().vidx[0] = vidx[0];
    faces_.back().vidx[1] = vidx[1];
    faces_.back().vidx[2] = vidx[2];

    return true;
  }

  void AddFace(Vertex const &v0, Vertex const &v1, Vertex const &v2) noexcept {
    faces_.emplace_back();
    vertices_.emplace_back(v0);
    faces_.back().vidx[0] = vertices_.size();
    vertices_.emplace_back(v1);
    faces_.back().vidx[1] = vertices_.size();
    vertices_.emplace_back(v2);
    faces_.back().vidx[2] = vertices_.size();
  }

  bool SaveAsObj(std::string const &path) const noexcept {
    std::ofstream ofs{path};
    if (!ofs.is_open()) {
      return false;
    }

    for (auto const &v : vertices_) {
      ofs << "v"
          << " " << v.point.x() << " " << v.point.y() << " " << v.point.z()
          << " " << (int)v.color.x() << " " << (int)v.color.y() << " " << (int)v.color.z()
          << std::endl;
    }
    for (auto const &f : faces_) {
      ofs << "f"
          << " " << f.vidx[0] << " " << f.vidx[1] << " " << f.vidx[2]
          << std::endl;
    }
    ofs.close();

    return true;
  }

  void Clear() noexcept {
    vertices_.clear();
    faces_.clear();
  }

  // std::vector<Vertex> &vertices() { return vertices_{}; }

  std::vector<Vertex> const &vertices() const { return vertices_; }

  // std::vector<Face> &faces() { return faces_{}; }

  std::vector<Face> const &faces() const { return faces_; }

private:
  std::vector<Vertex> vertices_{};
  std::vector<Face> faces_{};
};

typedef TriangleMesh<float> TriangleMeshf;
typedef TriangleMesh<double> TriangleMeshd;

} // namespace geometry