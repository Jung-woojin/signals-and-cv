import argparse
from statistics import mean

import numpy as np

from _signal_utils import (
    blobs,
    conv2d_reflect,
    gaussian_kernel,
    psnr,
    save_json,
    sobel_edges,
)


def add_weather_like_degradation(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # fog-like veiling + blur
    fog = 0.2 + 0.8 * x
    k = gaussian_kernel(size=7, sigma=1.6)
    blur = conv2d_reflect(fog, k)

    # rain-like streaks
    h, w = x.shape
    rain = np.zeros_like(x)
    for _ in range(80):
        y = int(rng.integers(0, h))
        x0 = int(rng.integers(0, w))
        length = int(rng.integers(6, 14))
        for t in range(length):
            yy = min(h - 1, y + t)
            xx = min(w - 1, x0 + t // 2)
            rain[yy, xx] += 0.25

    noise = 0.05 * rng.normal(size=x.shape).astype(np.float32)
    out = blur + rain + noise
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32)


def denoise(x: np.ndarray) -> np.ndarray:
    k = gaussian_kernel(size=5, sigma=1.0)
    return conv2d_reflect(x, k)


def sharpen(x: np.ndarray) -> np.ndarray:
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return conv2d_reflect(x, k)


def run(output_dir: str, n_samples: int) -> None:
    rows = []
    for i in range(n_samples):
        clean = blobs(128, 128, n_blobs=8, seed=2000 + i)
        degraded = add_weather_like_degradation(clean, seed=3000 + i)

        p_none = degraded
        p_denoise = denoise(degraded)
        p_dn_sharp = sharpen(p_denoise)

        e_clean = sobel_edges(clean).mean()
        rows.append(
            {
                "psnr_none": psnr(clean, p_none),
                "psnr_denoise": psnr(clean, p_denoise),
                "psnr_dn_sharp": psnr(clean, p_dn_sharp),
                "edge_ratio_none": float(sobel_edges(p_none).mean() / (e_clean + 1e-8)),
                "edge_ratio_denoise": float(sobel_edges(p_denoise).mean() / (e_clean + 1e-8)),
                "edge_ratio_dn_sharp": float(sobel_edges(p_dn_sharp).mean() / (e_clean + 1e-8)),
            }
        )

    result = {
        "n_samples": n_samples,
        "mean_metrics": {
            "psnr_none": float(mean(r["psnr_none"] for r in rows)),
            "psnr_denoise": float(mean(r["psnr_denoise"] for r in rows)),
            "psnr_dn_sharp": float(mean(r["psnr_dn_sharp"] for r in rows)),
            "edge_ratio_none": float(mean(r["edge_ratio_none"] for r in rows)),
            "edge_ratio_denoise": float(mean(r["edge_ratio_denoise"] for r in rows)),
            "edge_ratio_dn_sharp": float(mean(r["edge_ratio_dn_sharp"] for r in rows)),
        },
        "interpretation": {
            "note": "denoise는 PSNR을 올리지만 edge를 약화할 수 있고, denoise+sharpen은 edge를 일부 회복하지만 노이즈 재증폭 위험이 있다."
        },
    }

    save_json(f"{output_dir}/robust_perception_toy_ablation.json", result)
    print("[done] saved:", f"{output_dir}/robust_perception_toy_ablation.json")
    print(result["mean_metrics"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--n-samples", type=int, default=40)
    args = parser.parse_args()
    run(args.output_dir, args.n_samples)


if __name__ == "__main__":
    main()
