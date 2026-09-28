"""Combine the restarted 50/20, 100/20, and 120/20 pixel curves."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_session(folder: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((folder / "length_analysis_summary_corrected.json").read_text(
        encoding="utf-8"))
    with (folder / "length_samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_two", type=Path)
    parser.add_argument("last", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    first, first_rows = read_session(args.first_two)
    last, last_rows = read_session(args.last)
    stages = first["stages"] + last["stages"]
    if [s["label"] for s in stages] != ["Q1_50", "Q1_100", "Q1_120"]:
        raise RuntimeError("expected the three restarted 50/20, 100/20, 120/20 stages")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for source, rows, offset in (("first_two", first_rows, 0),
                                 ("last", last_rows, 2)):
        for row in rows:
            all_rows.append({"stage": int(row["stage"]) + offset,
                             "setting": ("50/20", "100/20", "120/20")[
                                 int(row["stage"]) + offset - 1],
                             "source_session": source,
                             "elapsed_s": float(row["elapsed_s"]),
                             "capture_monotonic": float(row["capture_monotonic"]),
                             "source_frame": int(row["source_frame"]),
                             "hardware_frame_id": int(row["hardware_frame_id"]),
                             "interval_source": row["interval_source"],
                             "plug_count": int(row["plug_count"]),
                             "median_length_px": (float(row["median_length_px"])
                                                  if row["median_length_px"] else None)})
    with (out / "length_samples_50_100_120.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    total_rows = [{**row, "cumulative_run_s":
                   (row["stage"] - 1) * 300 + row["elapsed_s"]}
                  for row in all_rows]
    with (out / "length_samples_total.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(total_rows[0]))
        writer.writeheader()
        writer.writerows(total_rows)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.5), sharex=True)
    colors = ["#1d6fa5", "#d55e00", "#3d8b62"]
    for index, stage in enumerate(stages):
        stage_rows = [r for r in all_rows if r["stage"] == index + 1]
        valid = [r for r in stage_rows if r["median_length_px"] is not None]
        x = np.asarray([r["elapsed_s"] for r in valid])
        y = np.asarray([r["median_length_px"] for r in valid])
        ax = axes[index]
        ax.scatter(x, y, s=8, alpha=0.35, color=colors[index], label="Observed")
        grid = np.arange(300, dtype=float)
        trend = np.asarray([
            float(np.median(y[np.abs(x - t) <= 7.5]))
            if np.count_nonzero(np.abs(x - t) <= 7.5) >= 3 else np.nan
            for t in grid])
        ax.plot(grid, trend, lw=2, color=colors[index], label="15 s median")
        ax.set_xlim(0, 300)
        ax.set_ylabel("Length (px)")
        ax.set_title(f'{stage["label"].replace("Q1_", "")}/20 uL/min · '
                     f'{stage["valid_samples"]}/{stage["samples"]} valid samples')
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Time since each flow setting began (s)")
    fig.suptitle("Dark plug length during three 5-minute flow settings\n"
                 "Rectified image pixels; gaps are unmeasured periods")
    fig.tight_layout()
    chart = out / "length_vs_time_50_100_120.png"
    fig.savefig(chart, dpi=170)
    plt.close(fig)
    # One cumulative run clock joins all three stages. The third stage was
    # started after a refill, so connections over absent measurements are
    # dashed; their slope is not a measured transient.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    stage_points = []
    stage_trends = []
    for index in range(3):
        valid = [row for row in total_rows if row["stage"] == index + 1
                 and row["median_length_px"] is not None]
        x = np.asarray([row["cumulative_run_s"] for row in valid])
        y = np.asarray([row["median_length_px"] for row in valid])
        grid = np.arange(index * 300, (index + 1) * 300, dtype=float)
        trend = np.asarray([
            float(np.median(y[np.abs(x - t) <= 7.5]))
            if np.count_nonzero(np.abs(x - t) <= 7.5) >= 3 else np.nan
            for t in grid])
        stage_points.append((x, y))
        stage_trends.append((grid, trend))
    for zoom in (False, True):
        fig, ax = plt.subplots(figsize=(14, 6))
        stage_colors = ["#2d78ab", "#d98633", "#3d9871"]
        for index, ((x, y), (grid, trend)) in enumerate(zip(stage_points, stage_trends)):
            ax.axvspan(index * 300, (index + 1) * 300,
                       color=stage_colors[index], alpha=0.045)
            ax.scatter(x, y, s=10, alpha=0.33, color=stage_colors[index],
                       label="观测点" if index == 0 else None)
            ax.plot(grid, trend, color="#142b43", lw=2,
                    label="15 秒中位线" if index == 0 else None)
        finite = [(float(t), float(value))
                  for grid, trend in stage_trends for t, value in zip(grid, trend)
                  if np.isfinite(value)]
        gap_labeled = False
        for (x0, y0), (x1, y1) in zip(finite, finite[1:]):
            if x1 - x0 > 1.5:
                ax.plot((x0, x1), (y0, y1), color="#65717d", ls="--", lw=1.2,
                        label="未测量区间的连接" if not gap_labeled else None)
                gap_labeled = True
        for boundary in (300, 600):
            ax.axvline(boundary, color="#606975", lw=1, ls=":")
        for index, label in enumerate(("50/20", "100/20", "120/20")):
            ax.text(index * 300 + 150, 0.985, label + " μL/min",
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    color=stage_colors[index], fontsize=12, weight="bold")
        ax.set_xlim(0, 900)
        ax.set_xticks(np.arange(0, 901, 100))
        ax.set_ylim(75, 112 if zoom else 210)
        ax.set_xlabel("累计运行时间（秒；每段 300 秒）")
        ax.set_ylabel("深色液柱长度（扶正图像素）")
        ax.set_title("三段阶跃实验：液柱长度总变化" + ("（局部放大）" if zoom else ""))
        ax.grid(alpha=0.18)
        ax.text(0.5, -0.18,
                "600 秒处为补液后单独启动；虚线只连接无有效测量的区间，不代表实际变化轨迹。",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                color="#4a5561")
        if zoom:
            ax.text(0.01, 0.08, "早期 >112 px 的少量观测点超出此放大视图",
                    transform=ax.transAxes, fontsize=9, color="#505d68")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.90),
                  ncol=3, fontsize=8, framealpha=0.85)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out / ("length_vs_time_total_zoom.png" if zoom
                           else "length_vs_time_total.png"), dpi=170)
        plt.close(fig)
    report = {"chart": str(chart),
              "total_chart": str(out / "length_vs_time_total.png"),
              "total_zoom_chart": str(out / "length_vs_time_total_zoom.png"),
              "stages": stages,
              "length_unit": "rectified image pixels",
              "physical_scale_validated": False,
              "sampling_premise_ok": bool(first["sampling_premise_ok"]
                                          and last["sampling_premise_ok"]),
              "note": "Separate last-stage session after refill; stage clocks each reset to zero."}
    (out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
