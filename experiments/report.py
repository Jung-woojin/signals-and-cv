import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, Optional


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_ablation_scripts(python_exe: str) -> None:
    scripts = [
        "experiments/anti_aliasing_stride_ablation.py",
        "experiments/small_object_aliasing_ablation.py",
        "experiments/robust_perception_toy_ablation.py",
    ]
    for s in scripts:
        print(f"[run] {s}")
        subprocess.run([python_exe, s], check=True)


def summarize_anti_aliasing(data: Dict[str, Any]) -> Dict[str, Any]:
    patterns = data.get("patterns", {})
    rows = []
    for name, row in patterns.items():
        n = float(row["shift_consistency_mae"]["naive"])
        a = float(row["shift_consistency_mae"]["antialias"])
        improve = ((n - a) / (n + 1e-12)) * 100.0
        rows.append(
            {
                "pattern": name,
                "naive_consistency_mae": n,
                "antialias_consistency_mae": a,
                "improvement_percent": improve,
            }
        )

    mean_improve = sum(r["improvement_percent"] for r in rows) / max(len(rows), 1)
    return {"rows": rows, "mean_improvement_percent": mean_improve}


def summarize_small_object(data: Dict[str, Any]) -> Dict[str, Any]:
    mos = data.get("mean_object_score", {})
    sss = data.get("shift_sensitivity_std", {})
    n_score = float(mos.get("naive", 0.0))
    a_score = float(mos.get("antialias", 0.0))
    n_shift = float(sss.get("naive", 0.0))
    a_shift = float(sss.get("antialias", 0.0))
    return {
        "mean_object_score": {"naive": n_score, "antialias": a_score},
        "shift_sensitivity_std": {"naive": n_shift, "antialias": a_shift},
        "better_stability": "antialias" if a_shift < n_shift else "naive",
    }


def summarize_robust(data: Dict[str, Any]) -> Dict[str, Any]:
    mm = data.get("mean_metrics", {})
    return {
        "psnr_none": float(mm.get("psnr_none", 0.0)),
        "psnr_denoise": float(mm.get("psnr_denoise", 0.0)),
        "psnr_dn_sharp": float(mm.get("psnr_dn_sharp", 0.0)),
        "edge_ratio_none": float(mm.get("edge_ratio_none", 0.0)),
        "edge_ratio_denoise": float(mm.get("edge_ratio_denoise", 0.0)),
        "edge_ratio_dn_sharp": float(mm.get("edge_ratio_dn_sharp", 0.0)),
        "interpretation": data.get("interpretation", {}).get("note", ""),
    }


def markdown_report(
    anti: Optional[Dict[str, Any]], sobj: Optional[Dict[str, Any]], robust: Optional[Dict[str, Any]]
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("# Signal-aware CV Toy Ablation Report")
    lines.append("")
    lines.append(f"- 생성 시각: {now}")
    lines.append("")

    lines.append("## 1) Anti-aliasing vs Naive Stride")
    if anti is None:
        lines.append("- 결과 파일 없음: `anti_aliasing_stride_ablation.json`")
    else:
        lines.append(
            f"- 평균 일관성 개선율(naive 대비 antialias): **{anti['mean_improvement_percent']:.2f}%**"
        )
        lines.append("")
        lines.append("| pattern | naive_consistency_mae | antialias_consistency_mae | improvement(%) |")
        lines.append("|---|---:|---:|---:|")
        for r in anti["rows"]:
            lines.append(
                f"| {r['pattern']} | {r['naive_consistency_mae']:.5f} | "
                f"{r['antialias_consistency_mae']:.5f} | {r['improvement_percent']:.2f} |"
            )

    lines.append("")
    lines.append("## 2) Small-object Aliasing Sensitivity")
    if sobj is None:
        lines.append("- 결과 파일 없음: `small_object_aliasing_ablation.json`")
    else:
        m = sobj["mean_object_score"]
        s = sobj["shift_sensitivity_std"]
        lines.append(
            f"- mean object score: naive={m['naive']:.4f}, antialias={m['antialias']:.4f}"
        )
        lines.append(
            f"- shift sensitivity std: naive={s['naive']:.5f}, antialias={s['antialias']:.5f}"
        )
        lines.append(f"- 더 안정적인 설정: **{sobj['better_stability']}**")

    lines.append("")
    lines.append("## 3) Robust Perception Toy (Weather/Noise/Blur)")
    if robust is None:
        lines.append("- 결과 파일 없음: `robust_perception_toy_ablation.json`")
    else:
        lines.append(
            f"- PSNR: none={robust['psnr_none']:.3f}, denoise={robust['psnr_denoise']:.3f}, "
            f"denoise+sharpen={robust['psnr_dn_sharp']:.3f}"
        )
        lines.append(
            f"- Edge ratio: none={robust['edge_ratio_none']:.3f}, denoise={robust['edge_ratio_denoise']:.3f}, "
            f"denoise+sharpen={robust['edge_ratio_dn_sharp']:.3f}"
        )
        if robust["interpretation"]:
            lines.append(f"- 해석: {robust['interpretation']}")

    lines.append("")
    lines.append("## 4) Quick Takeaways")
    lines.append("- anti-aliasing은 보통 shift-consistency를 개선한다.")
    lines.append("- 소물체 시나리오에서는 anti-aliasing이 shift 민감도를 낮출 가능성이 크다.")
    lines.append("- denoise는 안정성을 주지만 edge 손실이 발생할 수 있어 보강(sharpen/edge-aware loss) 전략이 필요하다.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate toy ablation JSONs and generate a report."
    )
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--output-dir", default="experiments/reports")
    parser.add_argument("--output-md", default="ablation_report.md")
    parser.add_argument("--output-json", default="ablation_report.json")
    parser.add_argument("--run-ablation", action="store_true")
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()

    if args.run_ablation:
        run_ablation_scripts(args.python_exe)

    anti_raw = load_json(os.path.join(args.results_dir, "anti_aliasing_stride_ablation.json"))
    sobj_raw = load_json(os.path.join(args.results_dir, "small_object_aliasing_ablation.json"))
    robust_raw = load_json(os.path.join(args.results_dir, "robust_perception_toy_ablation.json"))

    anti = summarize_anti_aliasing(anti_raw) if anti_raw else None
    sobj = summarize_small_object(sobj_raw) if sobj_raw else None
    robust = summarize_robust(robust_raw) if robust_raw else None

    report_md = markdown_report(anti, sobj, robust)
    summary_obj = {
        "generated_at": datetime.now().isoformat(),
        "anti_aliasing": anti,
        "small_object": sobj,
        "robust_perception": robust,
    }

    md_path = os.path.join(args.output_dir, args.output_md)
    json_path = os.path.join(args.output_dir, args.output_json)
    save_text(md_path, report_md)
    save_json(json_path, summary_obj)

    print(f"[done] markdown: {md_path}")
    print(f"[done] json: {json_path}")


if __name__ == "__main__":
    main()
