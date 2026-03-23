import argparse
from statistics import mean

import numpy as np

from _signal_utils import downsample_antialias, downsample_naive, make_small_objects, save_json, shift_image


def object_score_map(x: np.ndarray, method, factor: int) -> np.ndarray:
    y = method(x, factor=factor)
    y = y - y.min()
    y = y / (y.max() + 1e-8)
    return y


def run(output_dir: str, n_samples: int, factor: int) -> None:
    rng = np.random.default_rng(2026)
    naive_scores = []
    aa_scores = []
    naive_shift_std = []
    aa_shift_std = []

    for i in range(n_samples):
        img, coords = make_small_objects(128, 128, n_obj=5, seed=1000 + i)
        img = img + 0.03 * rng.normal(size=img.shape).astype(np.float32)

        shifts = [(0, 0), (0, 1), (1, 0), (1, 1)]
        per_shift_naive = []
        per_shift_aa = []

        for dy, dx in shifts:
            s_img = shift_image(img, dy, dx)
            nmap = object_score_map(s_img, downsample_naive, factor)
            amap = object_score_map(s_img, downsample_antialias, factor)

            nvals = []
            avals = []
            for (y, x) in coords:
                yy = min((y + dy) // factor, nmap.shape[0] - 1)
                xx = min((x + dx) // factor, nmap.shape[1] - 1)
                nvals.append(float(nmap[yy, xx]))
                avals.append(float(amap[yy, xx]))

            per_shift_naive.append(float(np.mean(nvals)))
            per_shift_aa.append(float(np.mean(avals)))

        naive_scores.append(float(np.mean(per_shift_naive)))
        aa_scores.append(float(np.mean(per_shift_aa)))
        naive_shift_std.append(float(np.std(per_shift_naive)))
        aa_shift_std.append(float(np.std(per_shift_aa)))

    result = {
        "factor": factor,
        "n_samples": n_samples,
        "mean_object_score": {
            "naive": float(mean(naive_scores)),
            "antialias": float(mean(aa_scores)),
        },
        "shift_sensitivity_std": {
            "naive": float(mean(naive_shift_std)),
            "antialias": float(mean(aa_shift_std)),
        },
        "summary": {
            "better_shift_stability": "antialias"
            if mean(aa_shift_std) < mean(naive_shift_std)
            else "naive"
        },
    }
    save_json(f"{output_dir}/small_object_aliasing_ablation.json", result)
    print("[done] saved:", f"{output_dir}/small_object_aliasing_ablation.json")
    print("mean object score (naive/aa):", result["mean_object_score"])
    print("shift sensitivity std (naive/aa):", result["shift_sensitivity_std"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()
    run(args.output_dir, args.n_samples, args.factor)


if __name__ == "__main__":
    main()
