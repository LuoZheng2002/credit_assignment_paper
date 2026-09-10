#!/usr/bin/env python3
"""Generate the branch-budget sensitivity chart for the TreeMAPPO paper."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def add_upper_headroom(axis, values: list[float], *, fraction: float = 0.35) -> None:
    lower, upper = axis.get_ylim()
    data_min = min(values)
    data_max = max(values)
    span = max(data_max - data_min, upper - lower, 1.0)
    axis.set_ylim(lower, data_max + span * fraction)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/qwen25_branch_budget_sensitivity.csv"))
    parser.add_argument("--output", type=Path, default=Path("images/qwen25_branch_budget_sensitivity.pdf"))
    args = parser.parse_args()

    rows = sorted(read_rows(args.input), key=lambda row: int(row["leaves"]))
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    import matplotlib.pyplot as plt

    leaves = [int(row["leaves"]) for row in rows]
    validation_gains = [float(row["validation_gain"]) * 100.0 for row in rows]
    test_macros = [float(row["test_macro"]) * 100.0 for row in rows]

    plt.rcParams.update({
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, left_axis = plt.subplots(figsize=(5.8, 3.0))
    right_axis = left_axis.twinx()

    validation_line = left_axis.plot(
        leaves,
        validation_gains,
        marker="o",
        linewidth=1.8,
        markersize=5,
        color="#1f77b4",
        label="Best validation gain",
    )
    test_line = right_axis.plot(
        leaves,
        test_macros,
        marker="s",
        linewidth=1.8,
        markersize=5,
        color="#d62728",
        label="Held-out test macro",
    )

    left_axis.set_xlabel("Total trajectories / leaves per tree")
    left_axis.set_ylabel("Best validation gain (points)", color="#1f77b4")
    right_axis.set_ylabel("Held-out test macro accuracy (%)", color="#d62728")
    left_axis.tick_params(axis="y", labelcolor="#1f77b4")
    right_axis.tick_params(axis="y", labelcolor="#d62728")
    left_axis.set_xticks(leaves)
    left_axis.grid(axis="y", alpha=0.25, linewidth=0.7)

    lines = validation_line + test_line
    labels = [line.get_label() for line in lines]
    add_upper_headroom(left_axis, validation_gains)
    add_upper_headroom(right_axis, test_macros)
    left_axis.legend(
        lines,
        labels,
        frameon=True,
        framealpha=0.82,
        edgecolor="none",
        fontsize=7,
        handlelength=1.4,
        loc="upper right",
    )
    left_axis.set_title("Qwen2.5-7B no-tool branch-budget sensitivity")

    for row, x, y in zip(rows, leaves, test_macros):
        right_axis.annotate(
            f"e{row['test_epoch']}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8,
            color="#d62728",
        )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
