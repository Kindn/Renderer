import numpy as np
import cv2
from typing import Tuple
import sys
from pythonperlin import perlin


def generate_random_float_map(size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
  return np.random.uniform(0.0, 1.0, size=size)


def generate_sine_texture(random_float_map: np.ndarray,
                          period: float = 32.0) -> np.ndarray:
  rows, cols = random_float_map.shape
  half_rows, half_cols = rows / 2, cols / 2
  ri, ro = min(half_rows, half_cols) / 2, min(half_rows, half_cols)

  def f(i, j):
    x, y = i - half_cols, j - half_rows
    r = np.sqrt(x**2 + y**2)
    return np.where((r >= ri) & (r <= ro), 0.5 * ((r - ro) / (ri - ro) * np.sin(2.0 * np.pi * r / period) + (r - ro) / (ri - ro)), 0.0)

  prob_map = np.fromfunction(f, (rows, cols), dtype=np.float64)

  texture = np.zeros_like(random_float_map, dtype=np.uint8)
  texture[random_float_map <= prob_map] = 255

  return texture


def generate_sine_density_texture(size: Tuple[int, int] = (1024, 1024),
                                  period: float = 32.0) -> np.ndarray:
  rows, cols = size
  half_rows, half_cols = rows / 2, cols / 2
  ri, ro = min(half_rows, half_cols) / 2, min(half_rows, half_cols)

  def f(i, j):
    x, y = i - half_cols, j - half_rows
    r = np.sqrt(x**2 + y**2)
    return np.where((r >= ri) & (r <= ro),
                    0.5 * ((r - ro) / (ri - ro) * np.sin(2.0 *
                           np.pi * r / period) + (r - ro) / (ri - ro)),
                    0.0)

  density_map = np.fromfunction(f, (rows, cols), dtype=np.float64)

  texture = (density_map * 255.0).astype(dtype=np.uint8)

  return texture


def generate_power_density_texture(size: Tuple[int, int] = (1024, 1024),
                                   gamma: float = 0.5) -> np.ndarray:
  rows, cols = size
  half_rows, half_cols = rows / 2, cols / 2
  ri, ro = min(half_rows, half_cols) / 2, min(half_rows, half_cols)

  def f(i, j):
    x, y = i - half_cols, j - half_rows
    r = np.sqrt(x**2 + y**2)
    return np.where((r >= ri) & (r <= ro), ((r - ro) / (ri - ro))**gamma, 0.0)

  density_map = np.fromfunction(f, (rows, cols), dtype=np.float64)

  texture = (density_map * 255.0).astype(dtype=np.uint8)

  return texture


def generate_multi_ring_density_texture(size: Tuple[int, int] = (1024, 1024),
                                        num_rings: int = 64) -> np.ndarray:
  rows, cols = size
  half_rows, half_cols = rows / 2, cols / 2
  ri, ro = min(half_rows, half_cols) / 2, min(half_rows, half_cols)

  ring_widths = np.random.dirichlet(
      alpha=np.ones(num_rings), size=1) * (ro - ri)
  cum_widths = np.cumsum(ring_widths)
  ring_densities = np.sqrt(np.random.uniform(0.2, 1.0, num_rings + 1))
  ring_densities = 1.0 / (1.0 + np.exp(-10.0 * (ring_densities - 0.5)))
  ring_densities[-1] = 0.0
  ring_densities[-2] = 0.0

  def f(i, j):
    x, y = i - half_cols, j - half_rows
    r = np.sqrt(x**2 + y**2)
    indices = np.searchsorted(cum_widths + ri, r, side="right")
    return ring_densities[np.where((indices < num_rings) & (r >= ri), indices, num_rings)]

  density_map = np.fromfunction(f, (rows, cols), dtype=np.float64)

  def f_edge(i, j):
    x, y = i - half_cols, j - half_rows
    r = np.sqrt(x**2 + y**2)
    return np.where(r > cum_widths[-2], ((r - ro) / (ri - ro))**0.5, 0.0)

  edge_density = np.fromfunction(f_edge, (rows, cols), dtype=np.float64)
  density_map += edge_density

  texture = (density_map * 255.0).astype(dtype=np.uint8)

  return texture


def generate_fbm_density_texture():
  shape = (64, 16)
  dens = 32
  noise_map = perlin(shape, dens=dens, seed=42, octaves=4)
  min_noise = np.min(noise_map)
  max_noise = np.max(noise_map)
  density_map = (noise_map - min_noise) / (max_noise - min_noise)
  
  texture = (density_map * 255.0).astype(dtype=np.uint8)
  texture = cv2.resize(texture, (4096, 4096))

  return texture


if __name__ == "__main__":
  texture_generators = {"sine": generate_sine_texture,
                        "sine_density": generate_sine_density_texture,
                        "power_density": generate_power_density_texture,
                        "multi_ring_density": generate_multi_ring_density_texture,
                        "fbm_density": generate_fbm_density_texture}

  tex_type = sys.argv[1]
  if tex_type not in texture_generators.keys():
    raise f"Invalid texture type {tex_type}"
  # random_float_map = generate_random_float_map()
  # texture = generate_sine_texture(random_float_map=random_float_map)
  texture = texture_generators[tex_type]()
  texture = cv2.GaussianBlur(texture, (7, 7), sigmaX=2.0)
  cv2.imshow("texture", texture)
  cv2.waitKey()

  output_path = "./texture.png"
  if len(sys.argv) > 2:
    output_path = sys.argv[2]
  cv2.imwrite(output_path, texture)
