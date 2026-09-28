"""Join separately stopped 100/10, 100/30, 100/40 pixel-length runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SETTINGS = ("100/10", "100/30", "100/40")
LABELS = ("Q2_10", "Q2_30", "Q2_40")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("sessions", type=Path, nargs="+")
    args = parser.parse_args()
    if not 1 <= len(args.sessions) <= 3:
        raise ValueError("expected one to three ordered Q2 step sessions")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    combined: list[dict] = []
    stage_summaries = []
    for index, folder in enumerate(args.sessions):
        source = folder.resolve()
        run = json.loads((source / "session_summary.json").read_text(encoding="utf-8"))
        analysis = json.loads((source / "length_analysis_summary_corrected.json").read_text(
            encoding="utf-8"))
        if (not run["result_layers"]["run_completed"] or not run["stop"]["verified"]
                or len(analysis["stages"]) != 1
                or analysis["stages"][0]["label"] != LABELS[index]):
            raise RuntimeError(f"incomplete or out-of-order session: {source}")
        stage_summaries.append(analysis["stages"][0])
        with (source / "length_samples.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                elapsed = float(row["elapsed_s"])
                combined.append({"stage": index + 1, "setting_ul_min": SETTINGS[index],
                                 "source_session": source.name,
                                 "stage_elapsed_s": elapsed,
                                 "cumulative_running_s": index * 300.0 + elapsed,
                                 "median_length_px": row["median_length_px"],
                                 "plug_count": row["plug_count"],
                                 "interval_source": row["interval_source"],
                                 "hardware_frame_id": row["hardware_frame_id"]})
    with (output / "length_samples_total.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ("#2576aa", "#dd8530", "#3a946b")
    ends = []
    for index in range(len(args.sessions)):
        rows = [r for r in combined if r["stage"] == index + 1
                and r["median_length_px"]]
        x = np.asarray([float(r["cumulative_running_s"]) for r in rows])
        y = np.asarray([float(r["median_length_px"]) for r in rows])
        grid = index * 300 + np.arange(300, dtype=float)
        trend = np.asarray([
            float(np.median(y[np.abs(x - t) <= 7.5]))
            if np.count_nonzero(np.abs(x - t) <= 7.5) >= 3 else np.nan
            for t in grid])
        ax.axvspan(index * 300, (index + 1) * 300, color=colors[index], alpha=.045)
        ax.scatter(x, y, s=9, alpha=.24, color=colors[index])
        ax.plot(grid, trend, color=colors[index], linewidth=2.2,
                label=f"{SETTINGS[index]} µL/min · {len(rows)}/{stage_summaries[index]['samples']}")
        finite = np.flatnonzero(np.isfinite(trend))
        if finite.size:
            ends.append(((float(grid[finite[0]]), float(trend[finite[0]])),
                         (float(grid[finite[-1]]), float(trend[finite[-1]]))))
        if index:
            ax.axvline(index * 300, color="#667684", linestyle=":", linewidth=1)
    for prior, current in zip(ends, ends[1:]):
        (x0, y0), (x1, y1) = prior[1], current[0]
        ax.plot((x0, x1), (y0, y1), linestyle="--", color="#667684", linewidth=1)
    ax.set_xlim(0, len(args.sessions) * 300)
    ax.set_xlabel("累计运行时间（秒；每段 300 秒）")
    ax.set_ylabel("液柱轴向长度（扶正图像素）")
    ax.set_title(" → ".join(SETTINGS[:len(args.sessions)]) + "：液柱长度总变化")
    ax.grid(alpha=.18)
    ax.legend(loc="upper right", fontsize=9)
    fig.text(.5, .015,
             "各段之间停泵并重新启动；虚线跨越未测量的停机时段，不代表实际变化轨迹。"
             " 像素标尺未独立验证；采集有帧号缺口。",
             ha="center", fontsize=9, color="#4a5561")
    fig.tight_layout(rect=(0, .035, 1, 1))
    chart = output / "length_vs_time_total.png"
    fig.savefig(chart, dpi=170)
    plt.close(fig)
    report = {"chart": str(chart), "stages": stage_summaries,
              "physical_scale_validated": False,
              "sampling_premise_ok": False,
              "time_axis": "cumulative pump-running seconds; stopped intervals omitted"}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
