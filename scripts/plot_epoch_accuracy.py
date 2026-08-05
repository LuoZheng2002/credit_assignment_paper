#!/usr/bin/env python3
"""Generate epoch-accuracy line charts for TreeMAPPO paper experiments."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/qwen25_r32_lr1e6_epoch_accuracy.csv"))
    parser.add_argument("--output", type=Path, default=Path("images/qwen25_notool_epoch_accuracy.pdf"))
    parser.add_argument("--scenario", default="notool")
    parser.add_argument("--model", default="qwen25")
    parser.add_argument(
        "--metric",
        choices=("accuracy", "improvement"),
        default="accuracy",
        help="Plot absolute validation accuracy or improvement over each method's epoch-0 baseline.",
    )
    args = parser.parse_args()

    rows = [
        row
        for row in read_rows(args.input)
        if row["scenario"] == args.scenario and row["model"] == args.model
    ]
    if not rows:
        raise SystemExit(f"No rows found for scenario={args.scenario!r}, model={args.model!r}")

    import matplotlib.pyplot as plt

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        label = f"{row['method']} (r={row['lora_rank']}, lr={row['learning_rate']})"
        series[label].append((int(row["epoch"]), float(row["accuracy"])))

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    markers = ["o", "s", "^", "D"]
    for index, (label, points) in enumerate(sorted(series.items())):
        points.sort()
        baseline = next((accuracy for epoch, accuracy in points if epoch == 0), None)
        if args.metric == "improvement" and baseline is None:
            raise SystemExit(f"Series {label!r} has no epoch-0 baseline for improvement plot")
        xs = [epoch for epoch, _ in points]
        if args.metric == "accuracy":
            ys = [accuracy * 100.0 for _, accuracy in points]
        else:
            ys = [(accuracy - baseline) * 100.0 for _, accuracy in points]
        ax.plot(xs, ys, marker=markers[index % len(markers)], linewidth=1.8, markersize=4.5, label=label)
        best_index = max(range(len(points)), key=lambda i: points[i][1])
        ax.scatter([xs[best_index]], [ys[best_index]], s=52, facecolors="none", edgecolors="black", linewidths=1.0, zorder=4)

    ax.set_xlabel("Training epoch")
    if args.metric == "accuracy":
        ax.set_ylabel("Validation accuracy (%)")
        ax.set_title("Qwen2.5-7B no-tool validation accuracy")
    else:
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("Validation accuracy improvement (points)")
        ax.set_title("Qwen2.5-7B no-tool validation improvement")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    ax.set_xticks(sorted({int(row["epoch"]) for row in rows}))
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
