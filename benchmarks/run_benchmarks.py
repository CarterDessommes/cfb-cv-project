#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_VIDEO = ROOT / "test media/videos/bijon_run.mp4"
DEFAULT_FRAMES = [0, 60, 120, 180, 240, 295]
DEFAULT_KEY_FRAME_DIR = ROOT / "benchmarks/fixtures/key_frames/bijon_run"
DEFAULT_LABELS = ROOT / "benchmarks/fixtures/bijon_run_labels.json"
DEFAULT_BASELINE = ROOT / "benchmarks/baseline.json"
DEFAULT_RESULTS_DIR = ROOT / "benchmarks/results"
LABELING_DOC = ROOT / "benchmarks/fixtures/LABELING.md"

BENCHMARKS = {
    "bijon_run": {
        "video": DEFAULT_VIDEO,
        "frames": DEFAULT_FRAMES,
        "key_frame_dir": DEFAULT_KEY_FRAME_DIR,
        "labels": DEFAULT_LABELS,
        "baseline": DEFAULT_BASELINE,
        "results_dir": DEFAULT_RESULTS_DIR / "bijon_run",
    },
    "pass1": {
        "video": ROOT / "test media/videos/pass1.mp4",
        "frames": [0, 60, 120, 180, 240, 272],
        "key_frame_dir": ROOT / "benchmarks/fixtures/key_frames/pass1",
        "labels": ROOT / "benchmarks/fixtures/pass1_labels.json",
        "baseline": ROOT / "benchmarks/pass1_baseline.json",
        "results_dir": DEFAULT_RESULTS_DIR / "pass1",
    },
}

THRESHOLDS = {
    "fps_regression_pct": 0.10,
    "p95_regression_pct": 0.15,
    "player_f1_drop": 0.02,
    "team_accuracy_drop": 0.03,
    "jersey_accuracy_drop": 0.05,
    "ball_hit_rate_drop": 0.05,
}

run_pipeline_benchmark = None


def parse_frames(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def display_path(path: Path) -> str:
    if path.is_absolute() and path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


def write_labeling_doc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """# Benchmark Label Format

Label the PNGs in `key_frames/bijon_run/`, then save the labels in
`benchmarks/fixtures/bijon_run_labels.json`.

Player boxes are required for detection accuracy. Team, jersey number, and ball
labels are optional; metrics for missing optional labels are skipped.

```json
{
  "video_path": "test media/videos/bijon_run.mp4",
  "frames": [
    {
      "frame_number": 1,
      "image_path": "benchmarks/fixtures/key_frames/bijon_run/frame_000001.png",
      "players": [
        {
          "bbox": [100, 200, 150, 310],
          "team": "offense",
          "number": "12"
        }
      ],
      "ball": {
        "center": [850, 420]
      }
    }
  ]
}
```

Use pixel coordinates in the exported PNG's original resolution. Omit `team`,
`number`, or `ball` when you do not want that frame to count for the metric.
""",
    )


def label_template(video_path: Path, frames: list[int], key_frame_dir: Path) -> dict[str, Any]:
    return {
        "video_path": display_path(video_path),
        "frames": [
            {
                "frame_number": frame,
                "image_path": display_path(key_frame_dir / f"frame_{frame:06d}.png"),
                "players": [],
            }
            for frame in frames
        ],
    }


def image_path_for_frame(key_frame_dir: Path, frame_number: int, image: dict[str, Any] | None = None) -> str:
    original_name = image.get("extra", {}).get("name") if image else None
    if original_name:
        return display_path(key_frame_dir / original_name)
    return display_path(key_frame_dir / f"frame_{frame_number:06d}.png")


def extract_key_frames(video_path: Path, frames: list[int], output_dir: Path, labels_path: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    written: list[Path] = []
    for frame in frames:
        if frame < 0:
            raise ValueError(f"Frame numbers are zero-based and must be non-negative: {frame}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame} from {video_path}")
        out_path = output_dir / f"frame_{frame:06d}.png"
        if not cv2.imwrite(str(out_path), image):
            raise RuntimeError(f"Could not write key frame: {out_path}")
        written.append(out_path)
    cap.release()

    write_labeling_doc(LABELING_DOC)
    if not labels_path.exists():
        write_json(labels_path, label_template(video_path, frames, output_dir))
    return written


def _frame_number_from_image(image: dict[str, Any]) -> int:
    name = image.get("extra", {}).get("name") or image.get("file_name", "")
    match = re.search(r"frame_(\d+)", name)
    if not match:
        raise ValueError(f"Could not parse frame number from image name: {name}")
    return int(match.group(1))


def import_coco_labels(coco_path: Path, labels_path: Path, video_path: Path, key_frame_dir: Path) -> dict[str, Any]:
    data = read_json(coco_path)
    categories = {
        int(category["id"]): str(category["name"]).strip().lower()
        for category in data.get("categories", [])
    }
    images = {int(image["id"]): image for image in data.get("images", [])}
    frames: dict[int, dict[str, Any]] = {}

    for image in images.values():
        frame_number = _frame_number_from_image(image)
        frames[frame_number] = {
            "frame_number": frame_number,
            "image_path": image_path_for_frame(key_frame_dir, frame_number, image),
            "players": [],
        }

    ball_candidates: dict[int, list[tuple[float, dict[str, Any]]]] = {}
    for annotation in data.get("annotations", []):
        image = images[int(annotation["image_id"])]
        frame_number = _frame_number_from_image(image)
        category = categories.get(int(annotation["category_id"]), "")
        x, y, width, height = [float(value) for value in annotation["bbox"]]

        if category in {"offense", "defense"}:
            frames[frame_number]["players"].append({
                "bbox": [round(x), round(y), round(x + width), round(y + height)],
                "team": category,
            })
        elif category == "player":
            frames[frame_number]["players"].append({
                "bbox": [round(x), round(y), round(x + width), round(y + height)],
            })
        elif category == "ball":
            ball_candidates.setdefault(frame_number, []).append((
                width * height,
                {"center": [round(x + width / 2), round(y + height / 2)]},
            ))

    for frame_number, candidates in ball_candidates.items():
        candidates.sort(key=lambda item: item[0], reverse=True)
        frames[frame_number]["ball"] = candidates[0][1]

    labels = {
        "video_path": display_path(video_path),
        "frames": [frames[frame_number] for frame_number in sorted(frames)],
    }
    write_json(labels_path, labels)
    return labels


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_players(labels: list[dict[str, Any]], predictions: list[dict[str, Any]], iou_threshold: float = 0.5):
    candidates = []
    for label_idx, label in enumerate(labels):
        for pred_idx, prediction in enumerate(predictions):
            iou = bbox_iou(label["bbox"], prediction["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, label_idx, pred_idx))
    candidates.sort(reverse=True)

    matched_labels: set[int] = set()
    matched_predictions: set[int] = set()
    matches = []
    for iou, label_idx, pred_idx in candidates:
        if label_idx in matched_labels or pred_idx in matched_predictions:
            continue
        matched_labels.add(label_idx)
        matched_predictions.add(pred_idx)
        matches.append((labels[label_idx], predictions[pred_idx], iou))
    return matches


def _accuracy(correct: int, total: int) -> float | None:
    return correct / total if total else None


def compute_accuracy_metrics(labels_doc: dict[str, Any], predictions_doc: dict[str, Any]) -> dict[str, Any]:
    predictions_by_frame = {
        int(frame["frame_number"]): frame
        for frame in predictions_doc.get("predictions", [])
    }

    detection_tp = detection_fp = detection_fn = 0
    team_correct = team_total = 0
    jersey_correct = jersey_total = 0
    ball_hits = ball_total = 0

    for label_frame in labels_doc.get("frames", []):
        frame_number = int(label_frame["frame_number"])
        labels = label_frame.get("players", [])
        predictions = predictions_by_frame.get(frame_number, {}).get("players", [])
        matches = match_players(labels, predictions)

        detection_tp += len(matches)
        detection_fp += max(0, len(predictions) - len(matches))
        detection_fn += max(0, len(labels) - len(matches))

        for label, prediction, _iou in matches:
            team = label.get("team")
            if team in {"offense", "defense"}:
                team_total += 1
                if prediction.get("team") == team:
                    team_correct += 1

            number = label.get("number")
            if number not in (None, ""):
                jersey_total += 1
                if str(prediction.get("number")) == str(number):
                    jersey_correct += 1

        ball = label_frame.get("ball")
        if ball and "center" in ball:
            ball_total += 1
            predicted_ball = predictions_by_frame.get(frame_number, {}).get("ball")
            if predicted_ball and "center" in predicted_ball:
                lx, ly = ball["center"]
                px, py = predicted_ball["center"]
                if math.hypot(float(px) - float(lx), float(py) - float(ly)) <= 20.0:
                    ball_hits += 1

    precision = detection_tp / (detection_tp + detection_fp) if detection_tp + detection_fp else 0.0
    recall = detection_tp / (detection_tp + detection_fn) if detection_tp + detection_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "player_precision": precision,
        "player_recall": recall,
        "player_f1": f1,
        "player_true_positives": detection_tp,
        "player_false_positives": detection_fp,
        "player_false_negatives": detection_fn,
        "team_accuracy": _accuracy(team_correct, team_total),
        "team_labeled_count": team_total,
        "jersey_accuracy": _accuracy(jersey_correct, jersey_total),
        "jersey_labeled_count": jersey_total,
        "ball_hit_rate": _accuracy(ball_hits, ball_total),
        "ball_labeled_count": ball_total,
    }


def _drop_too_large(current: float | None, baseline: float | None, allowed_drop: float) -> bool:
    if current is None or baseline is None:
        return False
    return current < baseline - allowed_drop


def _metric_improved(current: float | None, baseline: float | None, higher_is_better: bool) -> bool:
    if current is None or baseline is None:
        return False
    return current > baseline if higher_is_better else current < baseline


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    current_timings = current["timings"]
    baseline_timings = baseline["timings"]
    current_accuracy = current["accuracy"]
    baseline_accuracy = baseline["accuracy"]

    failures = []
    baseline_fps = baseline_timings["fps"]
    if baseline_fps > 0 and current_timings["fps"] < baseline_fps * (1.0 - THRESHOLDS["fps_regression_pct"]):
        failures.append(f"FPS regressed: {current_timings['fps']:.2f} < {baseline_fps:.2f}")

    baseline_p95 = baseline_timings["p95_frame_seconds"]
    if baseline_p95 > 0 and current_timings["p95_frame_seconds"] > baseline_p95 * (1.0 + THRESHOLDS["p95_regression_pct"]):
        failures.append(
            f"p95 frame time regressed: {current_timings['p95_frame_seconds']:.4f}s > {baseline_p95:.4f}s"
        )

    if _drop_too_large(current_accuracy["player_f1"], baseline_accuracy["player_f1"], THRESHOLDS["player_f1_drop"]):
        failures.append(
            f"player F1 regressed: {current_accuracy['player_f1']:.3f} < {baseline_accuracy['player_f1']:.3f}"
        )
    if _drop_too_large(current_accuracy["team_accuracy"], baseline_accuracy.get("team_accuracy"), THRESHOLDS["team_accuracy_drop"]):
        failures.append("team accuracy regressed")
    if _drop_too_large(current_accuracy["jersey_accuracy"], baseline_accuracy.get("jersey_accuracy"), THRESHOLDS["jersey_accuracy_drop"]):
        failures.append("jersey accuracy regressed")
    if _drop_too_large(current_accuracy["ball_hit_rate"], baseline_accuracy.get("ball_hit_rate"), THRESHOLDS["ball_hit_rate_drop"]):
        failures.append("ball hit rate regressed")

    improved = any([
        _metric_improved(current_timings["fps"], baseline_timings.get("fps"), True),
        _metric_improved(current_timings["p95_frame_seconds"], baseline_timings.get("p95_frame_seconds"), False),
        _metric_improved(current_accuracy["player_f1"], baseline_accuracy.get("player_f1"), True),
        _metric_improved(current_accuracy["team_accuracy"], baseline_accuracy.get("team_accuracy"), True),
        _metric_improved(current_accuracy["jersey_accuracy"], baseline_accuracy.get("jersey_accuracy"), True),
        _metric_improved(current_accuracy["ball_hit_rate"], baseline_accuracy.get("ball_hit_rate"), True),
    ])
    return len(failures) == 0, improved, failures


def print_summary(run: dict[str, Any]) -> None:
    timings = run["timings"]
    accuracy = run["accuracy"]
    config = run.get("config")
    if config:
        print(f"Config: imgsz={config['imgsz']} det_every={config['det_every']} "
              f"ocr_every={config['ocr_every']} ball_every={config['ball_every']}")
    print(f"FPS: {timings['fps']:.2f}")
    print(f"Mean frame: {timings['mean_frame_seconds']:.4f}s")
    print(f"p95 frame: {timings['p95_frame_seconds']:.4f}s")
    print(f"Player F1: {accuracy['player_f1']:.3f}")
    for key, label in [
        ("team_accuracy", "Team accuracy"),
        ("jersey_accuracy", "Jersey accuracy"),
        ("ball_hit_rate", "Ball hit rate"),
    ]:
        value = accuracy[key]
        print(f"{label}: {'skipped' if value is None else f'{value:.3f}'}")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    labels = read_json(args.labels)
    frames = [int(frame["frame_number"]) for frame in labels.get("frames", [])]
    if not frames:
        raise RuntimeError(f"No labeled frames found in {args.labels}")
    if not any(frame.get("players") for frame in labels.get("frames", [])):
        raise RuntimeError(
            f"No player labels found in {args.labels}. Run --get-key-frames, label the PNGs, "
            "then fill each frame's players list."
        )

    runner = run_pipeline_benchmark
    if runner is None:
        from pipeline import run_pipeline_benchmark as runner

    predictions = runner(
        video_path=str(args.video),
        frame_numbers=frames,
        det_model_path=str(args.det),
        ocr_model_path=str(args.ocr),
        ball_model_path=None if args.no_ball else str(args.ball),
        homography_path=str(args.homography),
        conf=args.conf,
        ocr_every=args.ocr_every,
        ball_every=args.ball_every,
        det_imgsz=args.imgsz,
        det_every=args.det_every,
    )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "video_path": str(args.video),
        "labels_path": str(args.labels),
        "thresholds": THRESHOLDS,
        "config": {
            "conf": args.conf,
            "imgsz": args.imgsz,
            "det_every": args.det_every,
            "ocr_every": args.ocr_every,
            "ball_every": args.ball_every,
            "no_ball": args.no_ball,
        },
        "timings": predictions["timings"],
        "accuracy": compute_accuracy_metrics(labels, predictions),
        "predictions": predictions["predictions"],
    }


def save_result(run: dict[str, Any], results_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"benchmark_{stamp}.json"
    write_json(path, run)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run labeled-frame speed and accuracy benchmarks.")
    parser.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="bijon_run")
    parser.add_argument("--get-key-frames", action="store_true", help="Export frames to label and create a label template.")
    parser.add_argument("--import-coco", type=Path, help="Convert a Roboflow/COCO annotation JSON into benchmark labels.")
    parser.add_argument("--check", action="store_true", help="Compare against baseline, fail on regressions, update on clean improvements.")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frames")
    parser.add_argument("--key-frame-dir", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--det", type=Path, default=ROOT / "weights/player-best.pt")
    parser.add_argument("--ocr", type=Path, default=ROOT / "weights/jersey_ocr.pt")
    parser.add_argument("--ball", type=Path, default=ROOT / "weights/ball-best.pt")
    parser.add_argument("--homography", type=Path, default=ROOT / "homographies.npz")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--det-every", type=int, default=1)
    parser.add_argument("--ocr-every", type=int, default=5)
    parser.add_argument("--ball-every", type=int, default=1)
    parser.add_argument("--no-ball", action="store_true")
    return parser


def apply_benchmark_defaults(args: argparse.Namespace) -> None:
    config = BENCHMARKS[args.benchmark]
    if args.video is None:
        args.video = config["video"]
    if args.frames is None:
        args.frames = ",".join(str(frame) for frame in config["frames"])
    if args.key_frame_dir is None:
        args.key_frame_dir = config["key_frame_dir"]
    if args.labels is None:
        args.labels = config["labels"]
    if args.baseline is None:
        args.baseline = config["baseline"]
    if args.results_dir is None:
        args.results_dir = config["results_dir"]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_benchmark_defaults(args)

    if args.get_key_frames:
        frames = parse_frames(args.frames)
        written = extract_key_frames(args.video, frames, args.key_frame_dir, args.labels)
        print(f"Wrote {len(written)} key frames to {args.key_frame_dir}")
        print(f"Label template: {args.labels}")
        print(f"Instructions: {LABELING_DOC}")
        return 0

    if args.import_coco:
        labels = import_coco_labels(args.import_coco, args.labels, args.video, args.key_frame_dir)
        player_count = sum(len(frame.get("players", [])) for frame in labels["frames"])
        ball_count = sum(1 for frame in labels["frames"] if frame.get("ball"))
        print(f"Wrote {len(labels['frames'])} labeled frames to {args.labels}")
        print(f"Players: {player_count}")
        print(f"Ball-labeled frames: {ball_count}")
        return 0

    try:
        run = run_benchmark(args)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 2
    result_path = save_result(run, args.results_dir)
    print_summary(run)
    print(f"Result: {result_path}")

    if not args.check:
        return 0

    if not args.baseline.exists():
        write_json(args.baseline, run)
        print(f"Created baseline: {args.baseline}")
        return 0

    passed, improved, failures = compare_to_baseline(run, read_json(args.baseline))
    if not passed:
        print("Benchmark check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if improved:
        write_json(args.baseline, run)
        print(f"Benchmark improved; updated baseline: {args.baseline}")
    else:
        print("Benchmark passed; baseline unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
