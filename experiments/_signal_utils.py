import json
import os
from typing import List, Tuple

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= np.sum(k)
    return k.astype(np.float32)


def conv2d_reflect(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = x.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw)), mode="reflect")
    y = np.zeros_like(x, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            patch = xp[i : i + kh, j : j + kw]
            y[i, j] = float(np.sum(patch * k))
    return y


def downsample_naive(x: np.ndarray, factor: int = 2) -> np.ndarray:
    return x[::factor, ::factor]


def downsample_antialias(x: np.ndarray, factor: int = 2) -> np.ndarray:
    sigma = max(0.8, factor * 0.5)
    k = gaussian_kernel(size=5, sigma=sigma)
    x_lp = conv2d_reflect(x, k)
    return x_lp[::factor, ::factor]


def shift_image(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    return np.roll(np.roll(x, dy, axis=0), dx, axis=1)


def checkerboard(h: int = 128, w: int = 128, tile: int = 4) -> np.ndarray:
    yy, xx = np.indices((h, w))
    return (((yy // tile + xx // tile) % 2).astype(np.float32) * 2.0 - 1.0)


def stripes(h: int = 128, w: int = 128, period: int = 6) -> np.ndarray:
    xx = np.arange(w)[None, :]
    img = np.sin(2 * np.pi * xx / period)
    return np.repeat(img, h, axis=0).astype(np.float32)


def blobs(h: int = 128, w: int = 128, n_blobs: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.indices((h, w))
    for _ in range(n_blobs):
        cy = rng.integers(0, h)
        cx = rng.integers(0, w)
        sigma = rng.uniform(4, 12)
        amp = rng.uniform(0.5, 1.0)
        img += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
    img -= img.mean()
    img /= img.std() + 1e-6
    return img.astype(np.float32)


def fft_highfreq_ratio(x: np.ndarray, cutoff_ratio: float = 0.25) -> float:
    f = np.fft.fftshift(np.fft.fft2(x))
    mag2 = np.abs(f) ** 2
    h, w = x.shape
    yy, xx = np.indices((h, w))
    cy, cx = h // 2, w // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = np.sqrt(cy**2 + cx**2) + 1e-8
    high = rr >= (cutoff_ratio * rmax)
    return float(mag2[high].sum() / (mag2.sum() + 1e-12))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def psnr(ref: np.ndarray, x: np.ndarray, data_range: float = 2.0) -> float:
    m = mse(ref, x)
    if m <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((data_range**2) / m))


def sobel_edges(x: np.ndarray) -> np.ndarray:
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gx = conv2d_reflect(x, kx)
    gy = conv2d_reflect(x, ky)
    return np.sqrt(gx**2 + gy**2)


def make_small_objects(h: int, w: int, n_obj: int, seed: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float32)
    coords: List[Tuple[int, int]] = []
    for _ in range(n_obj):
        y = int(rng.integers(4, h - 4))
        x = int(rng.integers(4, w - 4))
        r = int(rng.integers(1, 3))
        yy, xx = np.indices((h, w))
        spot = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * (r**2)))
        img += spot.astype(np.float32)
        coords.append((y, x))
    img /= img.max() + 1e-6
    return img, coords
