"""Plot a completed multi-stage capture on its actual acquisition clock."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    out = args.session_dir.resolve()
    session = json.loads((out / "session_summary.json").read_text(encoding="utf-8"))
    segments = session["segments"]
    analysis = json.loads((out / "length_analysis_summary_corrected.json").read_text(
        encoding="utf-8"))
    with (out / "length_samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(segments) != len(analysis["stages"]):
        raise RuntimeError("segment and analysis count differ")
    t0 = float(segments[0]["observation_started_monotonic"])
    data = [{"stage": int(row["stage"]),
             "actual_elapsed_s": float(row["capture_monotonic"]) - t0,
             "stage_elapsed_s": float(row["elapsed_s"]),
             "median_length_px": (float(row["median_length_px"])
                                  if row["median_length_px"] else None),
             "plug_count": int(row["plug_count"]),
             "hardware_frame_id": int(row["hardware_frame_id"]),
             "interval_source": row["interval_source"]} for row in rows]
    with (out / "length_samples_actual_time.csv").open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ("#2679ac", "#d47b2e", "#329667", "#7856a8")
    last_valid: tuple[float, float] | None = None
    stage_stats = []
    for index, segment in enumerate(segments):
        start = segment["observation_started_monotonic"] - t0
        end = segment["observation_finished_monotonic"] - t0
        color = colors[index % len(colors)]
        ax.axvspan(start, end, color=color, alpha=0.055)
        stage_rows = [row for row in data if row["stage"] == index + 1]
        valid = [row for row in stage_rows if row["median_length_px"] is not None]
        x = np.asarray([row["actual_elapsed_s"] for row in valid])
        y = np.asarray([row["median_length_px"] for row in valid])
        ax.scatter(x, y, s=9, color=color, alpha=0.35)
        grid = np.arange(np.ceil(start), np.floor(end) + 1, dtype=float)
        trend = np.asarray([
            float(np.median(y[np.abs(x - t) <= 7.5]))
            if np.count_nonzero(np.abs(x - t) <= 7.5) >= 3 else np.nan
            for t in grid])
        ax.plot(grid, trend, color=color, lw=2.0,
                label=(f'{int(segment["q1_ul_min"])}/{int(segment["q2_ul_min"])} '
                       f'μL/min · {len(valid)}/{len(stage_rows)} 有效点'))
        finite = [(float(t), float(value)) for t, value in zip(grid, trend)
                  if np.isfinite(value)]
        if last_valid is not None and finite:
            ax.plot((last_valid[0], finite[0][0]),
                    (last_valid[1], finite[0][1]),
                    ls="--", lw=1.0, color="#78818b")
        if finite:
            last_valid = finite[-1]
        if index:
            ax.axvline(start, ls=":", lw=1.0, color="#63707b")
        ax.text((start + end) / 2, 0.985,
                f'{int(segment["q1_ul_min"])}/{int(segment["q2_ul_min"])}',
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=12, weight="bold", color=color)
        stage_stats.append({"label": segment["label"],
                            "start_actual_elapsed_s": round(start, 3),
                            "end_actual_elapsed_s": round(end, 3),
                            "valid_samples": len(valid),
                            "all_samples": len(stage_rows),
                            "median_px": float(np.median(y)) if len(y) else None})
    ax.set_xlim(0, max(s["end_actual_elapsed_s"] for s in stage_stats))
    ax.set_xlabel("自第一段开始的实际采集时间（秒）")
    ax.set_ylabel("深色液柱长度（扶正图像素）")
    ax.set_title("阶跃实验液柱长度总变化")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    chart = out / "length_vs_time_total_actual.png"
    fig.savefig(chart, dpi=170)
    plt.close(fig)
    report = {"chart": str(chart), "stages": stage_stats,
              "unit": "rectified image pixels", "physical_scale_validated": False,
              "sampling_premise_ok": session["result_layers"]["sampling_premise_ok"],
              "note": "Dashed connections cross intervals with no accepted measurements. "
                      "No physical trajectory is inferred there."}
    (out / "length_total_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
