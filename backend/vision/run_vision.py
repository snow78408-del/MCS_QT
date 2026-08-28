from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .config import PipelineConfig, default_config
    from .pipeline import VisionPipeline
    from .preprocess.rotate_video import rotate_video
except ImportError:
    from config import PipelineConfig, default_config
    from pipeline import VisionPipeline
    from preprocess.rotate_video import rotate_video


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    config = default_config()
    config.debug.enabled = bool(args.debug)
    config.debug.verbose = bool(args.debug)

    config.tracker.tracker_type = args.tracker
    config.tracker.match_distance = float(args.match_distance)
    config.tracker.max_unmatched_frames = int(args.max_unmatched_frames)
    config.tracker.kalman.process_noise = float(args.kalman_process_noise)
    config.tracker.kalman.measurement_noise = float(args.kalman_measurement_noise)

    config.detector.min_radius = float(args.min_radius)
    config.detector.max_radius = float(args.max_radius)
    config.channel_region.enabled = not bool(args.skip_channel_region)
    config.channel_region.sample_frames = max(1, min(48, int(args.channel_region_sample_frames)))

    config.beads.mode = args.bead_mode
    config.beads.area_min = int(args.bead_area_min)
    config.beads.area_max = int(args.bead_area_max)

    if args.roi:
        x0, x1, y0, y1, crop_top = [float(x.strip()) for x in args.roi.split(",")]
        config.roi.enabled = True
        config.roi.user_defined = True
        config.roi.x_start_ratio = x0
        config.roi.x_end_ratio = x1
        config.roi.y_start_ratio = y0
        config.roi.y_end_ratio = y1
        config.roi.crop_top_ratio = crop_top

    return config


def preprocess_video_if_needed(args: argparse.Namespace) -> str:
    if args.preprocess_rotate == "none":
        return args.video

    output_path = args.preprocess_output
    if not output_path:
        src = Path(args.video)
        output_path = str(src.with_name(f"{src.stem}_preprocessed_{args.preprocess_rotate}.mp4"))

    rotate_video(args.video, output_path=output_path, mode=args.preprocess_rotate)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone vision pipeline.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str, help="Local video path")
    source.add_argument("--camera", type=int, help="Camera index")

    parser.add_argument("--output-video", type=str, default=None, help="Optional output annotated video path")
    parser.add_argument("--display", action="store_true", help="Show visualization window during processing")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only first N frames")
    parser.add_argument(
        "--skip-channel-region",
        action="store_true",
        help="Skip startup channel-region calibration and detect on the full/manual ROI frame",
    )
    parser.add_argument(
        "--channel-region-sample-frames",
        type=int,
        default=12,
        help="Startup frames used by automatic channel-region calibration",
    )

    parser.add_argument("--tracker", choices=["nearest", "kalman"], default="kalman")
    parser.add_argument("--bead-mode", choices=["intensity", "connected"], default="intensity")

    parser.add_argument("--min-radius", type=float, default=18.0)
    parser.add_argument("--max-radius", type=float, default=32.0)
    parser.add_argument("--match-distance", type=float, default=90.0)
    parser.add_argument("--max-unmatched-frames", type=int, default=8)

    parser.add_argument("--bead-area-min", type=int, default=5)
    parser.add_argument("--bead-area-max", type=int, default=80)

    parser.add_argument("--kalman-process-noise", type=float, default=8.0)
    parser.add_argument("--kalman-measurement-noise", type=float, default=12.0)

    parser.add_argument(
        "--preprocess-rotate",
        choices=["none", "ccw90", "cw90", "180", "auto"],
        default="none",
        help="Optional preprocess rotation before running vision",
    )
    parser.add_argument("--preprocess-output", type=str, default=None, help="Output path for preprocessed video")

    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI ratios as x0,x1,y0,y1,crop_top (e.g. 0.25,1.0,0.2,0.8,0.1)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = build_config_from_args(args)
    pipeline = VisionPipeline(config)

    if args.video:
        video_path = preprocess_video_if_needed(args)
        results = pipeline.process_video(
            video_path,
            output_path=args.output_video,
            display=args.display,
            max_frames=args.max_frames,
        )
    else:
        results = pipeline.process_camera(
            camera_index=args.camera,
            output_path=args.output_video,
            display=args.display,
            max_frames=args.max_frames,
        )

    if not results:
        print("No frames were processed.")
        return

    final_result = results[-1]
    print("Vision run complete.")
    print(f"Processed frames: {len(results)}")
    print(f"Total droplets: {final_result.metrics.analysis.total_droplets}")
    avg = final_result.metrics.control.average_diameter
    print(f"Average diameter: {avg:.3f}" if avg is not None else "Average diameter: None")
    print(f"Current active droplets: {final_result.metrics.control.frame_droplet_count}")
    print(f"New crossing count: {final_result.metrics.control.new_crossing_count}")
    print(f"Total real droplets: {final_result.metrics.control.total_droplet_count}")
    print(f"Valid for control: {final_result.metrics.control.valid_for_control}")
    print(f"Control validity reason: {final_result.metrics.control.reason}")
    print(f"Single-bead rate: {final_result.metrics.analysis.single_bead_rate:.2f}%")


if __name__ == "__main__":
    main()
