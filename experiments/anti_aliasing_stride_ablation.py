import argparse
from statistics import mean

import numpy as np

from _signal_utils import (
    blobs,
    checkerboard,
    downsample_antialias,
    downsample_naive,
    fft_highfreq_ratio,
    mae,
    save_json,
    shift_image,
    stripes,
)


def shift_consistency_error(img: np.ndarray, method, factor: int = 2) -> float:
    base = method(img, factor=factor)
    errors = []
    for dy in [0, 1, 2, 3]:
        for dx in [0, 1, 2, 3]:
            s_in = shift_image(img, dy, dx)
            s_out = method(s_in, factor=factor)
            expected = shift_image(base, dy // factor, dx // factor)
            errors.append(mae(s_out, expected))
    return float(mean(errors))


def run(output_dir: str, factor: int) -> None:
    patterns = {
        "checkerboard": checkerboard(128, 128, tile=4),
        "stripes": stripes(128, 128, period=6),
        "blobs": blobs(128, 128, n_blobs=8, seed=2026),
    }

    result = {"factor": factor, "patterns": {}}
    for name, img in patterns.items():
        naive = downsample_naive(img, factor=factor)
        aa = downsample_antialias(img, factor=factor)
        row = {
            "shift_consistency_mae": {
                "naive": shift_consistency_error(img, downsample_naive, factor),
                "antialias": shift_consistency_error(img, downsample_antialias, factor),
            },
            "highfreq_ratio": {
                "input": fft_highfreq_ratio(img),
                "naive": fft_highfreq_ratio(naive),
                "antialias": fft_highfreq_ratio(aa),
            },
            "summary": {
                "consistency_gain": "antialias better"
                if shift_consistency_error(img, downsample_antialias, factor)
                < shift_consistency_error(img, downsample_naive, factor)
                else "naive better",
            },
        }
        result["patterns"][name] = row

    save_json(f"{output_dir}/anti_aliasing_stride_ablation.json", result)
    print("[done] saved:", f"{output_dir}/anti_aliasing_stride_ablation.json")
    for name, row in result["patterns"].items():
        n = row["shift_consistency_mae"]["naive"]
        a = row["shift_consistency_mae"]["antialias"]
        print(f"{name:12s} | consistency mae naive={n:.5f} aa={a:.5f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--factor", type=int, default=2)
    args = parser.parse_args()
    run(args.output_dir, args.factor)


if __name__ == "__main__":
    main()
